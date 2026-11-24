"""Rotation, camera-ray, root-solve and fusion functions of ViDiHand2."""

import numpy as np
import torch
import torch.nn.functional as F
from scipy.linalg import solveh_banded

from baselines.vidihand_v2.mano import FINGERS, PALM


def rot6d_to_rotmat(x):
    shape = x.shape[:-1]
    x = x.float().reshape(-1, 2, 3).permute(0, 2, 1).contiguous()
    a1, a2 = x[:, :, 0], x[:, :, 1]
    b1 = F.normalize(a1, dim=-1)
    b2 = F.normalize(a2 - torch.einsum("bi,bi->b", b1, a2).unsqueeze(-1) * b1, dim=-1)
    return torch.stack((b1, b2, torch.linalg.cross(b1, b2, dim=-1)), dim=-1).reshape(*shape, 3, 3)


def ray_features(h, w, intr, canvas_wh, freq):
    fx, fy, cx, cy = intr
    i, j = torch.meshgrid(torch.arange(h, device=freq.device, dtype=torch.float32), torch.arange(w, device=freq.device, dtype=torch.float32),
                          indexing="ij")
    azimuth = torch.atan(((j + 0.5) * float(canvas_wh[0]) / w - cx) / fx).reshape(-1, 1)
    elevation = torch.atan(((i + 0.5) * float(canvas_wh[1]) / h - cy) / fy).reshape(-1, 1)
    f = freq.unsqueeze(0)
    return torch.cat([torch.sin(azimuth * f), torch.cos(azimuth * f), torch.sin(elevation * f), torch.cos(elevation * f)], -1)


BAND_U = 6


def edge_distance(uv, canvas_wh):
    w, h = (float(v) for v in canvas_wh)
    u, v = uv[..., 0], uv[..., 1]
    inside = torch.minimum(torch.minimum(u, w - u), torch.minimum(v, h - v))
    out_u = torch.clamp(-u, min=0) + torch.clamp(u - w, min=0)
    out_v = torch.clamp(-v, min=0) + torch.clamp(v - h, min=0)
    return torch.where((out_u > 0) | (out_v > 0), -torch.sqrt(out_u.square() + out_v.square()), inside)


def read_moments(mu, m2, canvas_wh):
    scale = torch.tensor([float(canvas_wh[0]), float(canvas_wh[1])], device=mu.device, dtype=torch.float32)
    mu, m2 = mu.float(), m2.float()
    return mu * scale, 0.5 * (scale.square() * (m2 - mu.square()).clamp_min(0.0)).sum(-1)


def assemble_reads(amodal_px, canvas_wh, fine, base_var, cfg):
    amodal_px = amodal_px.float()
    d_am = edge_distance(amodal_px, canvas_wh)
    var_am = (cfg.amodal_sigma0_px * (1.0 + (-d_am).clamp_min(0.0) / cfg.amodal_grow_px)).square()
    base_var = base_var.float()
    if not bool(torch.isfinite(base_var).all()):
        raise ValueError("ViDiHand2: non-finite token read variance")
    var_am = var_am + base_var.clamp_min(0.0)
    in_px, in_var, p_null, mass = (x.float() for x in fine)
    rel = (((cfg.fine_null[1] - p_null) / (cfg.fine_null[1] - cfg.fine_null[0])).clamp(0, 1)
           * ((mass - cfg.fine_mass[0]) / (cfg.fine_mass[1] - cfg.fine_mass[0])).clamp(0, 1))
    if not (torch.isfinite(in_px).all() and torch.isfinite(in_var).all()):
        raise ValueError("ViDiHand2: non-finite heat-map read")
    alpha = ((torch.minimum(d_am, edge_distance(in_px, canvas_wh)) - cfg.edge_px) / cfg.blend_px).clamp(0, 1) * rel
    delta2 = (in_px - amodal_px).square().sum(-1)
    read = alpha[..., None] * in_px + (1 - alpha[..., None]) * amodal_px
    base = torch.exp(alpha * torch.log(in_var.clamp_min(1.0)) + (1 - alpha) * torch.log(var_am.clamp_min(1.0)))
    var = base + alpha * (1 - alpha) * delta2 + alpha * (delta2 - cfg.disagree * (in_var + var_am)).clamp_min(0.0)
    return read, var.clamp_min(1.0), alpha


def read_stats(read_px, var_px, p_null, win_mass, canvas_wh, clamp):
    w, h = (float(v) for v in canvas_wh)
    r = read_px.float()
    if not bool(torch.isfinite(r).all()):
        raise ValueError("ViDiHand2: non-finite heat-map read")
    lg = lambda p: torch.logit(p.float().clamp(clamp, 1 - clamp))
    edge = torch.minimum(torch.minimum(r[..., 0], w - r[..., 0]), torch.minimum(r[..., 1], h - r[..., 1]))
    return torch.stack([lg(win_mass), lg(1 - p_null.float()), torch.log(var_px.float().clamp_min(1.0) / 16.0), (edge / 32.0).clamp(0.0, 4.0)], -1)


def camera(intr4):
    return intr4[:, None, None].unbind(-1)


def rays(read_px, intr4):
    fx, fy, cx, cy = camera(intr4)
    return torch.stack([(read_px[..., 0] - cx) / fx, (read_px[..., 1] - cy) / fy, torch.ones_like(read_px[..., 0])], -1)


def depth_pairs(parents, knuckles):
    bones = [(j, parents[j]) for j in range(1, len(parents))]
    across = [(a, b) for i, a in enumerate(knuckles) for b in knuckles[i + 1:]]
    wrist = [(j, 0) for j in range(1, len(parents)) if parents[j] != 0]
    c, p = zip(*bones, *across, *wrist)
    return torch.tensor(c), torch.tensor(p)


def second_difference_gram(retained):
    t = retained.shape[1] + 2
    v = torch.zeros(t - 2, t, dtype=torch.float64, device=retained.device)
    idx = torch.arange(t - 2, device=retained.device)
    v[idx, idx] = 1.0
    v[idx, idx + 1] = -2.0
    v[idx, idx + 2] = 1.0
    return torch.einsum("ct,nc,cu->ntu", v, retained.double(), v)


def second_difference_bands(c):
    n, k = c.shape
    t = k + 2
    z = c.new_zeros(n, 2)
    cp, c1, c2 = torch.cat((c, z), 1), torch.cat((z[:, :1], c, z[:, :1]), 1), torch.cat((z, c), 1)
    return cp + 4.0 * c1 + c2, -2.0 * cp[:, :t - 1] - 2.0 * c1[:, :t - 1], c


def banded_matvec(bands, x):
    d0, d1, d2 = bands
    y = d0 * x
    y[:, :-1] += d1 * x[:, 1:]
    y[:, 1:] += d1 * x[:, :-1]
    y[:, :-2] += d2 * x[:, 2:]
    y[:, 2:] += d2 * x[:, :-2]
    return y


def solve_banded_spd(upper, b):
    ab, bb = upper.cpu().numpy(), b.cpu().numpy()
    return torch.as_tensor(np.stack([solveh_banded(ab[i], bb[i], lower=False, check_finite=False) for i in range(bb.shape[0])]), device=b.device)


def read_consistency(rel, read_px, var_px, w, intr4, cfg):
    f64, dev = torch.float64, rel.device
    intr = intr4.to(f64)
    relc = torch.nan_to_num(rel.to(f64))
    q = rays(torch.nan_to_num(read_px.to(f64)), intr)
    var = torch.nan_to_num(var_px.to(f64), nan=1e8, posinf=1e8).clamp_min(1.0)
    ok = torch.isfinite(read_px).all(-1) & torch.isfinite(rel).all(-1) & (torch.nan_to_num(w.to(f64)) > cfg.present)
    c, p = torch.triu_indices(rel.shape[2], rel.shape[2], 1, device=dev)
    L = (relc[:, :, c] - relc[:, :, p]).norm(dim=-1)
    dz = relc[..., 2] - relc[..., :1, 2]
    dq = q[:, :, c] - q[:, :, p]
    e = dz[:, :, c, None] * q[:, :, c] - dz[:, :, p, None] * q[:, :, p]
    qa, qb, qc = (dq * dq).sum(-1), (dq * e).sum(-1), (e * e).sum(-1) - L ** 2
    disc = qb ** 2 - qa * qc
    zq = (-qb + disc.clamp_min(0).sqrt()) / qa.clamp_min(1e-30)
    zm = torch.nanmedian(torch.where(ok[:, :, c] & ok[:, :, p] & (qa > 0) & (disc >= 0) & (zq > 0), zq, torch.full_like(L, float("nan"))), dim=-1).values
    P = (zm[..., None] + dz)[..., None] * q
    T = torch.nanmedian(torch.where(ok[..., None], P - relc, torch.full_like(P, float("nan"))), dim=-2).values
    X = relc + T[..., None, :]
    valid = torch.isfinite(zm) & torch.isfinite(T).all(-1) & torch.where(ok, X[..., 2] > 0, torch.ones_like(ok)).all(-1)
    Z = torch.where(valid[..., None], X[..., 2], torch.ones_like(X[..., 2]))
    fx, fy, cx, cy = camera(intr)
    r = torch.nan_to_num(read_px.to(f64)) - torch.stack((fx * X[..., 0] / Z + cx, fy * X[..., 1] / Z + cy), -1)
    J = torch.zeros(X.shape[:-1] + (2, 3), dtype=f64, device=dev)
    J[..., 0, 0], J[..., 0, 2], J[..., 1, 1], J[..., 1, 2] = fx / Z, -fx * X[..., 0] / Z ** 2, fy / Z, -fy * X[..., 1] / Z ** 2
    sig = torch.as_tensor(np.asarray(cfg.sigma_rel, np.float64), device=dev)
    C = var[..., None, None] * torch.eye(2, dtype=f64, device=dev) + sig[:, None, None] ** 2 * (J @ J.transpose(-1, -2))
    s = torch.sqrt(torch.einsum("...i,...ij,...j->...", r, torch.linalg.inv(C), r).clamp_min(0))
    k = cfg.huber_k
    wt = torch.where(s <= k, torch.ones_like(s), (1 - ((s - k) / (cfg.cons_c - k)).clamp(max=1) ** 2) ** 2)
    omega = torch.where(valid[..., None] & ok, wt, torch.ones_like(wt))
    w0 = omega[..., 0:1]
    ray = torch.where(valid[..., None], w0 * q[:, :, 0] + (1 - w0) * T / T[..., 2:3].clamp_min(1e-9), torch.full_like(T, float("nan")))
    return omega, ray, torch.where(valid[..., None] & ok, s, torch.full_like(s, float("nan")))


def depth_roots(rel, read_px, var_px, w, intr4, fps, omega, ray, cfg):
    dev, f64, t = rel.device, torch.float64, rel.shape[0]
    intr = intr4.to(f64)
    fx, fy, _, _ = camera(intr)
    fpx2 = fx * fy
    relc = torch.nan_to_num(rel.to(f64))
    q = rays(torch.nan_to_num(read_px.to(f64)), intr)
    dz = relc[..., 2] - relc[..., :1, 2]
    var = torch.nan_to_num(var_px.to(f64), nan=1e8, posinf=1e8).clamp_min(1.0)
    wj = torch.nan_to_num(w.to(f64)).clamp(0, 1)
    wj = torch.where(torch.isfinite(read_px).all(-1) & torch.isfinite(rel).all(-1), wj, torch.zeros_like(wj))
    c, p = (x.to(dev) for x in cfg.pairs)
    L = (relc[:, :, c] - relc[:, :, p]).norm(dim=-1)
    dq = q[:, :, c] - q[:, :, p]
    e = dz[:, :, c, None] * q[:, :, c] - dz[:, :, p, None] * q[:, :, p]
    wp = wj * omega.to(f64)
    wb = torch.minimum(wp[:, :, c], wp[:, :, p])
    vb = var[:, :, c] + var[:, :, p]
    ev = ((wj[:, :, c] > cfg.present) & (wj[:, :, p] > cfg.present)).sum(-1) >= cfg.min_bones
    lam_c = (float(fps) ** 2 / cfg.depth_acceleration) ** 2 * cfg.couple
    lo = torch.where(wj > 0, -dz, torch.full_like(dz, -float("inf"))).amax(-1)
    reject = ev & ~torch.isfinite(ray).all(-1)
    a = (wb / vb * dq.norm(dim=-1).square()).sum(-1)
    num = (wb / vb * L * dq.norm(dim=-1)).sum(-1)
    reject = reject | (ev & ~(a > 0))
    z = num / a.clamp_min(1e-30)
    reject = reject | (ev & ~(z > lo))
    ev = ev & ~reject
    z = torch.where(ev, z, lo + 1.0)
    banded = t > cfg.banded_min_t
    if banded:
        bands = second_difference_bands((ev[:-2] & ev[1:-1] & ev[2:]).permute(1, 0).to(f64) * lam_c)
    else:
        gram = second_difference_gram((ev[:-2] & ev[1:-1] & ev[2:]).permute(1, 0)) * lam_c
    evf = ev.to(f64)
    sig = z[..., None] * vb.sqrt() / fpx2.sqrt()
    sig = torch.where(torch.isfinite(sig) & (sig > 0), sig, torch.ones_like(sig))
    hk = cfg.huber_k

    def rows(z):
        v = z[..., None, None] * dq + e
        m = v.norm(dim=-1).clamp_min(1e-09)
        return (m - L) / sig, (v * dq).sum(-1) / m / sig, z > lo

    def coupling(z):
        zt = z.permute(1, 0)
        gz = banded_matvec(bands, zt) if banded else (gram @ zt[..., None])[..., 0]
        return 0.5 * (zt * gz).sum(-1), gz

    def energy(z):
        s, _, inside = rows(z)
        a_ = s.abs()
        rho = torch.where(a_ <= hk, 0.5 * s.square(), hk * a_ - 0.5 * hk * hk)
        er = (wb * evf[..., None] * rho).sum(-1).sum(0)
        return torch.where((ev & ~inside).any(0), torch.full_like(er, float("inf")), er + coupling(z)[0])

    lam = torch.full((2,), 1e-3, dtype=f64, device=dev)
    E = energy(z)
    conv = torch.zeros(2, dtype=torch.bool, device=dev)
    for _ in range(cfg.depth_maxit):
        if bool(conv.all()):
            break
        s, js, _ = rows(z)
        a_ = s.abs()
        w_ = wb * evf[..., None] * torch.where(a_ <= hk, torch.ones_like(s), hk / a_.clamp_min(1e-12))
        H = (w_ * js.square()).sum(-1)
        _, gz = coupling(z)
        g = (w_ * js * s).sum(-1).permute(1, 0) + gz
        lam_t = lam[:, None]
        if banded:
            up = torch.zeros(2, 3, t, dtype=f64, device=dev)
            dg = H.permute(1, 0) + bands[0]
            up[:, 2], up[:, 1, 1:], up[:, 0, 2:] = dg + lam_t * (dg + 1e-12) + 1e-12, bands[1], bands[2]
            step = solve_banded_spd(up, g).permute(1, 0)
        else:
            A = torch.diag_embed(H.permute(1, 0)) + gram
            dg = torch.diagonal(A, dim1=-2, dim2=-1)
            step = torch.linalg.solve(A + torch.diag_embed(lam_t * (dg + 1e-12) + 1e-12), g[..., None])[..., 0].permute(1, 0)
        zn = torch.where(ev, z - step, z)
        En = energy(zn)
        acc = torch.isfinite(En) & (En < E) & ~conv
        z = torch.where(acc[None, :], zn, z)
        conv = conv | (acc & ((step.abs() * evf).amax(0) < cfg.depth_tol))
        E = torch.where(acc, En, E)
        lam = torch.where(acc, (lam / 10.0).clamp_min(1e-12), lam * 10.0)
        conv = conv | (lam > 1e12)
    keep = ev & torch.isfinite(z)
    return torch.where(keep[..., None], z[..., None] * ray.to(f64), torch.full_like(q[:, :, 0], float("nan"))).float()


def weighted_median(x, w):
    xs, idx = x.sort(-1)
    ws = w.gather(-1, idx)
    cw = ws.cumsum(-1)
    tot = cw[..., -1:]
    k = (cw >= 0.5 * tot).float().argmax(-1, keepdim=True)
    med = xs.gather(-1, k)[..., 0]
    return torch.where(tot[..., 0] > 0, med, torch.full_like(med, float("nan")))


def motion_cues(read, evid, ok, alpha, p_null, fps, pairs):
    dev, t = read.device, read.shape[0]
    u = read.float()
    e = torch.nan_to_num(evid.float(), nan=0.0).clamp(0, 1)
    fin = torch.isfinite(u).all(-1)
    wj = torch.where(fin, e, torch.zeros_like(e)) * ok[..., None].float()
    u0 = torch.nan_to_num(u)
    c, p = (x.to(dev) for x in pairs)
    lens = (u0[:, :, c] - u0[:, :, p]).norm(dim=-1)
    lt = weighted_median(lens, wj[:, :, c] * wj[:, :, p])
    good = ok & torch.isfinite(lt) & (lt > 0)
    phi = torch.zeros(t, 2, 9, device=dev)
    nb = good[2:] & good[1:-1] & good[:-2]
    du = (u0[2:] - u0[:-2]).norm(dim=-1) * (float(fps) / 2.0)
    d2u = (u0[2:] - 2 * u0[1:-1] + u0[:-2]).norm(dim=-1) * float(fps) ** 2
    wm = wj[2:] * wj[1:-1] * wj[:-2]
    l1 = torch.where(good[1:-1], lt[1:-1], torch.ones_like(lt[1:-1]))
    vs = weighted_median(du, wm) / l1
    ac = weighted_median(d2u, wm) / l1
    ll = torch.log(torch.where(good, lt, torch.ones_like(lt)))
    lr = (ll[2:] - ll[:-2]) * (float(fps) / 2.0)
    la = (ll[2:] - 2 * ll[1:-1] + ll[:-2]).abs() * float(fps) ** 2
    z = torch.zeros_like(vs)
    phi[1:-1, :, 0] = torch.where(nb, torch.log1p(torch.nan_to_num(vs).clamp_min(0)), z)
    phi[1:-1, :, 1] = torch.where(nb, torch.log1p(torch.nan_to_num(ac).clamp_min(0)), z)
    phi[1:-1, :, 2] = torch.where(nb, torch.sign(lr) * torch.log1p(lr.abs()), z)
    phi[1:-1, :, 3] = torch.where(nb, torch.log1p(la), z)
    em = torch.where(ok, (e * fin.float()).mean(-1), torch.zeros(t, 2, device=dev))
    phi[..., 4] = em
    pad = torch.cat([em[:1].expand(2, 2), em, em[-1:].expand(2, 2)], 0)
    phi[..., 5] = torch.stack([pad[i:i + t] for i in range(5)], 0).min(0).values
    phi[..., 6] = torch.where(ok, torch.nan_to_num(alpha.float()).clamp(0, 1).mean(-1), torch.zeros_like(em))
    phi[..., 7] = torch.where(ok, torch.nan_to_num(p_null.float()).clamp(0, 1).mean(-1), torch.zeros_like(em))
    phi[..., 8] = (~ok).float()
    if not bool(torch.isfinite(phi).all()):
        raise ValueError("ViDiHand2: a non-finite motion cue")
    return phi


def read_cues(read, var, evid, alpha, p_null, s, canvas_wh):
    dev = read.device
    u, v = read.float(), var.float()
    fin = torch.isfinite(u).all(-1) & torch.isfinite(v) & (v > 0)
    e = torch.nan_to_num(evid.float(), nan=0.0).clamp(0, 1)
    de = edge_distance(torch.nan_to_num(u), canvas_wh)
    psi = torch.zeros(u.shape[:-1] + (9,), device=dev)
    psi[..., 0] = torch.log(e.clamp_min(1e-6))
    psi[..., 1] = torch.log(torch.where(fin, v, torch.ones_like(v)).clamp_min(1e-6))
    psi[..., 2] = torch.nan_to_num(p_null.float(), nan=0.0).clamp(0, 1)
    psi[..., 3] = torch.sign(de) * torch.log1p(de.abs())
    psi[..., 4] = torch.nan_to_num(alpha.float(), nan=0.0).clamp(0, 1)
    s = s.float()
    psi[..., 6] = torch.where(torch.isfinite(s), torch.log1p(torch.nan_to_num(s, nan=0.0).clamp_min(0)), torch.zeros_like(s))
    nf = fin.float().sum(-1).clamp_min(1.0)
    hmean = lambda x: ((x * fin.float()).sum(-1) / nf)[..., None].expand_as(e)
    psi[..., 5] = hmean(psi[..., 4])
    psi[..., 7] = hmean(psi[..., 3])
    psi[..., 8] = hmean(e)
    return torch.nan_to_num(torch.where(fin[..., None], psi, torch.zeros_like(psi)))


def links(claim, adj):
    keep, t, n = np.zeros_like(adj), 0, len(adj)
    while t < n:
        e = t
        while e < n - 1 and adj[e]:
            e += 1
        c = np.flatnonzero(claim[t:e + 1])
        if len(c) >= 2:
            keep[t + c[0]:t + c[-1]] = True
        t = e + 1
    return adj & keep


def chains(link):
    same = np.zeros(len(link) - 1, bool)
    row = link[:-2] & link[1:-1]
    same[:-1] |= row
    same[1:] |= row
    return np.concatenate([[0], np.cumsum(~same)]).astype(int)


def huber(s, k):
    return np.where(s <= k, 0.5 * s * s, k * (s - 0.5 * k))


class HandSolve:
    def __init__(self, rel, mu, cread, w, k, link, sigma, lam, cfg):
        rel, mu, cread, w = (np.array(a, np.float64) for a in (rel, mu, cread, w))
        okf = np.isfinite(rel).all(-1).all(-1)
        okj = np.isfinite(mu).all(-1) & np.isfinite(cread) & (cread > 0) & okf[:, None] & np.isfinite(w)
        self.w = np.where(okj, w, 0.0)
        self.mu = np.where(np.isfinite(mu), mu, 0.0)
        self.cread = np.where(np.isfinite(cread) & (cread > 0), cread, 1.0)
        self.rel = np.where(np.isfinite(rel), rel, 0.0)
        self.k = tuple(k[:, i, None] for i in range(4))
        self.n, self.sigma, self.cfg = rel.shape[0], sigma, cfg
        c = (link[:-2] & link[1:-1]).astype(np.float64)
        self.lbands = [tuple(b[0].numpy() for b in second_difference_bands(torch.as_tensor(c * lam[:, a])[None])) for a in range(3)]
        fx, fy, cx, cy = self.k
        self.q = np.stack(((self.mu[..., 0] - cx) / fx, (self.mu[..., 1] - cy) / fy, np.ones_like(self.mu[..., 0])), -1)
        self.c, self.p = (np.asarray(x) for x in cfg.pairs)
        self.L = np.linalg.norm(self.rel[:, self.c] - self.rel[:, self.p], axis=-1)
        self.lhat = (self.rel[:, self.c] - self.rel[:, self.p]) / np.maximum(self.L[..., None], 1e-9)
        self.wb = self.w[:, self.c] * self.w[:, self.p]

    def _jac(self, T):
        fx, fy = self.k[:2]
        P = self.rel + T[:, None, :]
        Z = np.where(self.w > 0, P[..., 2], 1.0)
        J = np.zeros(P.shape[:-1] + (2, 3))
        J[..., 0, 0] = fx / Z
        J[..., 0, 2] = -fx * P[..., 0] / Z ** 2
        J[..., 1, 1] = fy / Z
        J[..., 1, 2] = -fy * P[..., 1] / Z ** 2
        return P, Z, J

    def covs(self, T):
        fx, fy = self.k[:2]
        _, _, J = self._jac(T)
        C = self.cread[..., None, None] * np.eye(2) + self.sigma[None, :, None, None] ** 2 * np.einsum("...ia,...ja->...ij", J, J)
        c, p = self.c, self.p
        dz, Zw = self.rel[..., 2], T[:, 2]
        Dc, Dp = Zw[:, None] + dz[:, c], Zw[:, None] + dz[:, p]
        v = Dc[..., None] * self.q[:, c] - Dp[..., None] * self.q[:, p]
        nb = v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)
        s_ = Dc ** 2 * self.cread[:, c] + Dp ** 2 * self.cread[:, p]
        rd = s_ * (nb[..., 0] ** 2 / fx ** 2 + nb[..., 1] ** 2 / fy ** 2)
        ez = np.array([0.0, 0.0, 1.0])
        ac = (nb * self.q[:, c]).sum(-1)[..., None] * ez - self.lhat
        ap = -(nb * self.q[:, p]).sum(-1)[..., None] * ez + self.lhat
        rd = rd + self.sigma[c] ** 2 * (ac ** 2).sum(-1) + self.sigma[p] ** 2 * (ap ** 2).sum(-1)
        return np.linalg.inv(C), 1.0 / np.maximum(rd, 1e-12)

    def resid(self, T):
        fx, fy, cx, cy = self.k
        P, Z, J = self._jac(T)
        r = np.stack((fx * P[..., 0] / Z + cx, fy * P[..., 1] / Z + cy), -1) - self.mu
        c, p = self.c, self.p
        dz, Zw = self.rel[..., 2], T[:, 2]
        v = (Zw[:, None] + dz[:, c])[..., None] * self.q[:, c] - (Zw[:, None] + dz[:, p])[..., None] * self.q[:, p]
        m = np.maximum(np.linalg.norm(v, axis=-1), 1e-9)
        return r, J, m - self.L, (v * (self.q[:, c] - self.q[:, p])).sum(-1) / m

    def _couple(self, x, a):
        return banded_matvec(tuple(b[None] for b in self.lbands[a]), x[None])[0]

    def inside(self, T):
        return bool(((T[:, 2:3] + self.rel[..., 2]) > 0)[self.w > 0].all())

    def objective(self, T, Cinv, ib):
        if not self.inside(T):
            return np.inf
        k = self.cfg.huber_k
        r, _, rb, _ = self.resid(T)
        s = np.sqrt(np.maximum(np.einsum("...i,...ij,...j->...", r, Cinv, r), 0.0))
        e = (self.w * huber(s, k)).sum() + (self.wb * huber(np.abs(rb) * np.sqrt(ib), k)).sum()
        return e + 0.5 * sum(T[:, a] @ self._couple(T[:, a], a) for a in range(3))

    def system(self, T, Cinv, ib):
        k = self.cfg.huber_k
        r, J, rb, jb = self.resid(T)
        s = np.sqrt(np.maximum(np.einsum("...i,...ij,...j->...", r, Cinv, r), 1e-24))
        om = self.w * np.minimum(1.0, k / s)
        sb = np.abs(rb) * np.sqrt(ib)
        omb = self.wb * np.minimum(1.0, k / np.maximum(sb, 1e-12)) * ib
        JW = np.einsum("tjia,tjib->tjab", J, Cinv)
        Nb = np.einsum("tj,tjab,tjbc->tac", om, JW, J)
        g = np.einsum("tj,tjab,tjb->ta", om, JW, r)
        Nb[:, 2, 2] += (omb * jb ** 2).sum(-1)
        g[:, 2] += (omb * jb * rb).sum(-1)
        for a in range(3):
            g[:, a] += self._couple(T[:, a], a)
        return Nb, g

    def dense(self, Nb):
        n = self.n
        N = np.zeros((3 * n, 3 * n))
        idx = np.arange(n) * 3
        for a in range(3):
            for b in range(3):
                N[idx + a, idx + b] += Nb[:, a, b]
            d0, d1, d2 = self.lbands[a]
            ia = idx + a
            N[ia, ia] += d0
            N[ia[:-1], ia[1:]] += d1
            N[ia[1:], ia[:-1]] += d1
            N[ia[:-2], ia[2:]] += d2
            N[ia[2:], ia[:-2]] += d2
        return N

    def banded(self, Nb):
        ab = np.zeros((BAND_U + 1, 3 * self.n))
        idx = np.arange(self.n) * 3
        for a in range(3):
            for b in range(a, 3):
                ab[BAND_U - (b - a), idx + b] += Nb[:, a, b]
            d0, d1, d2 = self.lbands[a]
            ab[BAND_U, idx + a] += d0
            ab[BAND_U - 3, idx[1:] + a] += d1
            ab[BAND_U - 6, idx[2:] + a] += d2
        return ab

    def fill_start(self, T0):
        T = np.array(T0, np.float64)
        miss = ~np.isfinite(T).all(-1)
        if not miss.any():
            return T
        T[miss] = 0.0
        for a in range(3):
            d0, d1, d2 = self.lbands[a]
            hx = self._couple(np.where(miss, 0.0, T[:, a]), a)
            ab = np.zeros((3, self.n))
            ab[2] = np.where(miss, d0, 1.0)
            ab[1, 1:] = np.where(miss[:-1] & miss[1:], d1, 0.0)
            ab[0, 2:] = np.where(miss[:-2] & miss[2:], d2, 0.0)
            T[:, a] = np.where(miss, solveh_banded(ab, np.where(miss, -hx, 0.0)), T[:, a])
        return T

    def solve(self, T0, use_banded):
        T = self.fill_start(T0)
        Cinv, ib = self.covs(T)
        e, lam = self.objective(T, Cinv, ib), 1e-3
        for _ in range(self.cfg.maxit if np.isfinite(e) else 0):
            Nb, g = self.system(T, Cinv, ib)
            M = self.banded(Nb) if use_banded else self.dense(Nb)
            if not (np.isfinite(M).all() and np.isfinite(g).all()):
                break
            dN = (M[BAND_U] if use_banded else np.diag(M)).copy()
            if float(np.max(np.abs(g.ravel()) / np.sqrt(np.maximum(dN, 1e-30)))) < self.cfg.gtol:
                break
            ok = False
            for _ in range(12):
                damp = lam * np.maximum(dN, 1e-12)
                try:
                    if use_banded:
                        A = M.copy()
                        A[BAND_U] += damp
                        stp = solveh_banded(A, g.ravel(), lower=False, check_finite=False)
                    else:
                        stp = np.linalg.solve(M + np.diag(damp), g.ravel())
                except np.linalg.LinAlgError:
                    lam *= 10.0
                    continue
                Tn = T - stp.reshape(self.n, 3)
                en = self.objective(Tn, Cinv, ib)
                if np.isfinite(en) and en < e:
                    ok = True
                    break
                lam *= 10.0
            if not ok:
                break
            T, e, lam = Tn, en, max(lam / 10.0, 1e-9)
        return T


def solve_roots(rel, read, var, alpha, evid, claim, intr4, fps, init, acc_sigma, read_logvar, read_sigma, cfg):
    t = rel.shape[0]
    k = intr4.double().cpu().numpy()
    np_ = lambda x: x.double().cpu().numpy()
    rlv, sigma, sig_acc = np_(read_logvar), np_(read_sigma), np_(acc_sigma)
    reln, mu, V, al, rt0 = np_(rel), np_(read), np_(var), np_(alpha), np_(init)
    e = np.clip(np.nan_to_num(np_(evid), nan=cfg.e_floor), cfg.e_floor, 1.0)
    with np.errstate(invalid="ignore", over="ignore"):
        cread = V * np.exp(-2.0 * al * np.log(e) + rlv)
    cl = claim.cpu().numpy()
    adj = np.ones(t, bool)
    adj[-1] = False
    okr = np.isfinite(mu).all(-1)
    use_banded = t > cfg.banded_min_t
    out, unplaced = np.full((t, 2, 3), np.nan), np.zeros((t, 2), bool)
    for h in range(2):
        sm = sig_acc[1:-1, h]
        lam = float(fps) ** 4 / np.stack([sm[:, 0], sm[:, 0], sm[:, 1]], -1) ** 2
        relz = np.nan_to_num(reln[:, h, :, 2], nan=0.0)
        rowj = okr[:, h] & np.isfinite(cread[:, h]) & (cread[:, h] > 0) & np.isfinite(reln[:, h]).all(-1).all(-1)[:, None]
        nread = rowj.sum(-1)
        front = lambda T_: ~rowj | ((T_[:, 2:3] + relz) > 0)
        with np.errstate(invalid="ignore"):
            usable = np.isfinite(rt0[:, h]).all(-1) & front(rt0[:, h]).all(-1)
        cl_h = cl[:, h].copy()
        while True:
            lk = links(cl_h, adj)
            ch = chains(lk)
            nc = int(ch.max()) + 1
            r = np.where(cl_h, nread, 0)
            size, held = np.bincount(ch, minlength=nc), np.bincount(ch, usable, nc)
            det = np.where(size == 1, np.bincount(ch, r, nc) >= 2, np.bincount(ch, np.minimum(2 * r, 3), nc) >= 6)
            placed = det & np.where(size == 1, held >= 1, held >= 2) & (np.bincount(ch, cl_h, nc) > 0)
            drop = cl_h & ~placed[ch]
            if drop.any():
                cl_h &= ~drop
                continue
            used = placed[ch]
            if not used.any():
                break
            w = np.repeat(cl_h[:, None], rel.shape[2], 1).astype(np.float64)
            hand = HandSolve(reln[:, h], mu[:, h], cread[:, h], w, k, lk, sigma, lam, cfg)
            T0 = np.where(usable[:, None], rt0[:, h], np.nan)
            anchor = np.where(used & usable)[0]
            T0 = np.where(used[:, None], T0, rt0[anchor[np.clip(np.searchsorted(anchor, np.arange(t)), 0, len(anchor) - 1)], h])
            T0 = hand.fill_start(T0)
            out_dom = cl_h & ~front(T0).all(-1)
            if out_dom.any():
                cl_h &= ~out_dom
                continue
            break
        unplaced[:, h] = cl[:, h] & ~cl_h
        if used.any():
            out[:, h] = np.where(used[:, None], hand.solve(T0, use_banded), np.nan)
    return torch.as_tensor(out, dtype=rel.dtype, device=rel.device), torch.as_tensor(unplaced, device=rel.device)


def rotvec(r):
    w = torch.stack([r[:, 2, 1] - r[:, 1, 2], r[:, 0, 2] - r[:, 2, 0], r[:, 1, 0] - r[:, 0, 1]], -1) / 2
    sn = w.norm(dim=-1).clamp_min(1e-12)
    cs = (r.diagonal(dim1=1, dim2=2).sum(-1) - 1) / 2
    out = w / sn[:, None] * torch.atan2(sn, cs)[:, None]
    half = ((w.norm(dim=-1) < 1e-6) & (cs < 0)).nonzero()[:, 0]
    bm = (r[half] + torch.eye(3, device=r.device, dtype=r.dtype)[None]) / 2
    k = bm.diagonal(dim1=1, dim2=2).argmax(-1)
    ax = bm[torch.arange(len(half), device=r.device), :, k]
    ax = ax / ax.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    ax = torch.where(ax.gather(1, k[:, None]) < 0, -ax, ax)
    ax = torch.where((ax * w[half]).sum(-1, keepdim=True) < 0, -ax, ax)
    return out.index_put((half,), ax * torch.atan2(w[half].norm(dim=-1), cs[half])[:, None])


def rotmat(w):
    a = w.norm(dim=-1).clamp_min(1e-12)
    ax = w / a[:, None]
    a = a[:, None, None]
    k = torch.zeros(len(w), 3, 3, device=w.device, dtype=w.dtype)
    k[:, 0, 1], k[:, 0, 2], k[:, 1, 0], k[:, 1, 2], k[:, 2, 0], k[:, 2, 1] = -ax[:, 2], ax[:, 1], ax[:, 2], -ax[:, 0], -ax[:, 1], ax[:, 0]
    return torch.eye(3, device=w.device, dtype=w.dtype)[None] + torch.sin(a) * k + (1 - torch.cos(a)) * (k @ k)


def kabsch(p, q):
    u, _, vt = torch.linalg.svd((p.transpose(1, 2) @ q).double())
    d = torch.where(torch.linalg.det(vt.transpose(1, 2) @ u.transpose(1, 2)) < 0, -1.0, 1.0).to(u.dtype)
    return (vt.transpose(1, 2) @ torch.diag_embed(torch.stack([torch.ones_like(d), torch.ones_like(d), d], -1)) @ u.transpose(1, 2)).to(p.dtype)


def rigid_fits(f, hh):
    f, hh = f.double(), hh.double()
    cf, ch = f.mean(1, keepdim=True), hh.mean(1, keepdim=True)
    a = f - cf
    r = kabsch(a, hh - ch)
    rig = a @ r.transpose(1, 2) + ch
    wf = torch.stack([rotvec(kabsch(rig[:, list(c)] - rig[:, m:m + 1], hh[:, list(c)] - hh[:, m:m + 1])) for m, *c in FINGERS], 1)
    return dict(cf=cf, ch=ch, a=a, w=rotvec(r), wf=wf, nr=hh - (a @ r.transpose(1, 2) + ch))


def fit_cues(fi):
    lg = lambda v: torch.log(v.clamp_min(1e-6))
    return torch.cat([lg((fi["ch"] - fi["cf"])[:, 0].norm(dim=-1))[:, None], lg(fi["w"].norm(dim=-1))[:, None],
                      lg(fi["nr"].square().sum(-1).mean(-1).sqrt())[:, None], lg(fi["wf"].norm(dim=-1)), lg(fi["cf"][:, 0, 2])[:, None]], -1)


def apply_parts(fi, sh):
    sh = {k: v.double() for k, v in sh.items()}
    x0 = fi["a"] @ rotmat(fi["w"]).transpose(1, 2) + fi["cf"] + sh["t"][:, None, None] * (fi["ch"] - fi["cf"])
    xs = {j: x0[:, j] for j in PALM}
    for k, (m, *jj) in enumerate(FINGERS):
        q, s = (sh["qt"], sh["st"]) if k == 4 else (sh["q"][:, k], sh["s"][:, k])
        y = x0[:, m:m + 1] + (x0[:, jj] - x0[:, m:m + 1]) @ rotmat(fi["wf"][:, k] * q[:, None]).transpose(1, 2)
        y = y + s[:, None, None] * (x0[:, jj] + fi["nr"][:, jj] - y)
        for c, j in enumerate(jj):
            xs[j] = y[:, c]
    return torch.stack([xs[j] for j in range(21)], 1)


def fuse(on_rays, fk, g, parts, part_in):
    t, h, j, _ = fk.shape
    okj = (torch.isfinite(on_rays).all(-1) & torch.isfinite(fk).all(-1))[..., None]
    i = torch.isfinite(fk).all(-1).all(-1).reshape(t * h).nonzero()[:, 0]
    fz = torch.nan_to_num(fk)
    hh = fz + torch.where(okj, g * (torch.nan_to_num(on_rays) - fz), torch.zeros_like(fz))
    fi = rigid_fits(fz.reshape(t * h, j, 3)[i], hh.reshape(t * h, j, 3)[i])
    sh = parts(part_in.reshape(t * h, -1)[i], fit_cues(fi))
    x = fk.reshape(t * h, j, 3).index_put((i,), apply_parts(fi, sh).to(fk.dtype)).reshape(t, h, j, 3)
    return torch.where(torch.isfinite(on_rays).all(-1).all(-1)[..., None, None], x, fk + g * (on_rays - fk))


def ray_joints(read_px, z, rel, intr4):
    q = rays(read_px.float(), intr4.float())
    relz = rel.float()[..., 2]
    d = z.float()[..., None] + relz - relz[..., :1]
    bad = (~(d > 0) & torch.isfinite(d) & torch.isfinite(q).all(-1)).any(-1, keepdim=True)
    return torch.where(bad, torch.full_like(d, float("nan")), d)[..., None] * q
