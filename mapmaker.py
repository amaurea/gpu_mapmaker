if __name__ == "__main__":
	# Do argparse first to give quick feedback on wrong arguments
	# without waiting for slow imports
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument("ifiles", nargs="+")
	parser.add_argument("area")
	parser.add_argument("odir")
	parser.add_argument("-p", "--prefix",  type=str, default=None)
	parser.add_argument("-v", "--verbose", action="count", default=1)
	parser.add_argument("-q", "--quiet",   action="count", default=0)
	args = parser.parse_args()

import numpy as np, os, time
from pixell import enmap, utils, mpi, bunch, fft, colors, memory
import so3g
from sotodlib import coords

def read_tod(fname):
	"""Read a tod file in the simple npz format we use"""
	res = bunch.Bunch()
	# Could do this in a loop, but we do it explicitly so we
	# can document which fields should be present.
	# Change ra,dec and x,y order to dec,ra and y,x, as agreed with Kendrick
	with np.load(fname) as f:
		res.dets         = f["dets"]                 # [ndet]
		res.point_offset = f["point_offset"][:,::-1] # [ndet,{y,x}]
		res.polangle     = f["polangle"]             # [ndet]
		bore = f["boresight"]
		res.ctime        = bore[0]                   # [nsamp]
		res.boresight    = bore[[2,1]]               # [{el,az},nsamp]
		res.tod          = f["tod"]                  # [ndet,nsamp]
		res.cuts         = mask2cuts(f["cuts"])
	return res

def mask2cuts(mask):
	return so3g.proj.ranges.RangesMatrix.from_mask(mask)

class Logger:
	def __init__(self, level=0, id=0, fmt="{id:3d} {t:6.2f} {mem:6.2f} {max:6.2f} {msg:s}"):
		self.level = level
		self.id    = id
		self.fmt   = fmt
		self.t0    = time.time()
	def print(self, message, level=0, id=None, color=None, end="\n"):
		if level > self.level: return
		if id is not None and id != self.id: return
		msg = self.fmt.format(id=self.id, t=(time.time()-self.t0)/60, mem=memory.current()/1024**3, max=memory.max()/1024**3, msg=message)
		if color is not None:
			msg = color + msg + colors.reset
		print(msg, end=end)

class PmatCut:
	"""Implementation of cuts-as-extra-degrees-of-freedom for a single obs."""
	def __init__(self, cuts, model=None, params={"resolution":100, "nmax":100}):
		self.cuts   = cuts
		self.model  = model or "full"
		self.params = params
		self.njunk  = so3g.process_cuts(self.cuts.ranges, "measure", self.model, self.params, None, None)
	def forward(self, tod, junk):
		"""Project from the cut parameter (junk) space for this scan to tod."""
		so3g.process_cuts(self.cuts.ranges, "insert", self.model, self.params, tod, junk)
	def backward(self, tod, junk):
		"""Project from tod to cut parameters (junk) for this scan."""
		so3g.process_cuts(self.cuts.ranges, "extract", self.model, self.params, tod, junk)
		self.clear(tod)
	def clear(self, tod):
		junk = np.empty(self.njunk, tod.dtype)
		so3g.process_cuts(self.cuts.ranges, "clear", self.model, self.params, tod, junk)

class ArrayZipper:
	def __init__(self, shape, dtype, comm=None):
		self.shape = shape
		self.ndof  = int(np.prod(shape))
		self.dtype = dtype
		self.comm  = comm
	def zip(self, arr):  return arr.reshape(-1)
	def unzip(self, x):  return x.reshape(self.shape).astype(self.dtype, copy=False)
	def dot(self, a, b):
		return np.sum(a*b) if self.comm is None else self.comm.allreduce(np.sum(a*b))

class MapZipper:
	def __init__(self, shape, wcs, dtype, comm=None):
		self.shape, self.wcs = shape, wcs
		self.ndof  = int(np.prod(shape))
		self.dtype = dtype
		self.comm  = comm
	def zip(self, map): return np.asarray(map.reshape(-1))
	def unzip(self, x): return enmap.ndmap(x.reshape(self.shape), self.wcs).astype(self.dtype, copy=False)
	def dot(self, a, b):
		return np.sum(a*b) if self.comm is None else utils.allreduce(np.sum(a*b),self.comm)

class MultiZipper:
	def __init__(self):
		self.zippers = []
		self.ndof	= 0
		self.bins	= []
	def add(self, zipper):
		self.zippers.append(zipper)
		self.bins.append([self.ndof, self.ndof+zipper.ndof])
		self.ndof += zipper.ndof
	def zip(self, *objs):
		return np.concatenate([zipper.zip(obj) for zipper, obj in zip(self.zippers, objs)])
	def unzip(self, x):
		res = []
		for zipper, (b1,b2) in zip(self.zippers, self.bins):
			res.append(zipper.unzip(x[b1:b2]))
		return res
	def dot(self, a, b):
		res = 0
		for (b1,b2), dof in zip(self.bins, self.zippers):
			res += dof.dot(a[b1:b2],b[b1:b2])
		return res

class Nmat:
	def __init__(self):
		"""Initialize the noise model. In subclasses this will typically set up parameters, but not
		build the details that depend on the actual time-ordered data"""
		self.ivar  = np.ones(1, dtype=np.float32)
		self.ready = True
	def build(self, tod, **kwargs):
		"""Measure the noise properties of the given time-ordered data tod[ndet,nsamp], and
		return a noise model object tailored for that specific tod. The returned object
		needs to provide the .apply(tod) method, which multiplies the tod by the inverse noise
		covariance matrix. Usually the returned object will be of the same class as the one
		we call .build(tod) on, just with more of the internal state initialized."""
		return self
	def apply(self, tod):
		"""Multiply the time-ordered data tod[ndet,nsamp] by the inverse noise covariance matrix.
		This is done in-pace, but the result is also returned."""
		return tod*self.ivar
	def white(self, tod):
		"""Like apply, but without detector or time correlations"""
		return tod*self.ivar
	def write(self, fname):
		bunch.write(fname, bunch.Bunch(type="Nmat"))
	@staticmethod
	def from_bunch(data): return Nmat()

class NmatUncorr(Nmat):
	def __init__(self, spacing="exp", nbin=100, nmin=10, window=2, bins=None, ips_binned=None, ivar=None, nwin=None):
		self.spacing    = spacing
		self.nbin       = nbin
		self.nmin       = nmin
		self.bins       = bins
		self.ips_binned = ips_binned
		self.ivar       = ivar
		self.window     = window
		self.nwin       = nwin
		self.ready      = bins is not None and ips_binned is not None and ivar is not None
	def build(self, tod, srate, **kwargs):
		# Apply window while taking fft
		nwin  = utils.nint(self.window*srate)
		apply_window(tod, nwin)
		ft    = fft.rfft(tod)
		# Unapply window again
		apply_window(tod, nwin, -1)
		ps = np.abs(ft)**2
		del ft
		if   self.spacing == "exp": bins = utils.expbin(ps.shape[-1], nbin=self.nbin, nmin=self.nmin)
		elif self.spacing == "lin": bins = utils.expbin(ps.shape[-1], nbin=self.nbin, nmin=self.nmin)
		else: raise ValueError("Unrecognized spacing '%s'" % str(self.spacing))
		ps_binned  = utils.bin_data(bins, ps) / tod.shape[1]
		ips_binned = 1/ps_binned
		# Compute the representative inverse variance per sample
		ivar = np.zeros(len(tod))
		for bi, b in enumerate(bins):
			ivar += ips_binned[:,bi]*(b[1]-b[0])
		ivar /= bins[-1,1]-bins[0,0]
		return NmatUncorr(spacing=self.spacing, nbin=len(bins), nmin=self.nmin, bins=bins, ips_binned=ips_binned, ivar=ivar, window=self.window, nwin=nwin)
	def apply(self, tod, inplace=False):
		if inplace: tod = np.array(tod)
		apply_window(tod, self.nwin)
		ftod = fft.rfft(tod)
		# Candidate for speedup in C
		norm = tod.shape[1]
		for bi, b in enumerate(self.bins):
			ftod[:,b[0]:b[1]] *= self.ips_binned[:,None,bi]/norm
		# I divided by the normalization above instead of passing normalize=True
		# here to reduce the number of operations needed
		fft.irfft(ftod, tod)
		apply_window(tod, self.nwin)
		return tod
	def white(self, tod, inplace=True):
		if not inplace: tod = np.array(tod)
		apply_window(tod, self.nwin)
		tod *= self.ivar[:,None]
		apply_window(tod, self.nwin)
		return tod
	def write(self, fname):
		data = bunch.Bunch(type="NmatUncorr")
		for field in ["spacing", "nbin", "nmin", "bins", "ips_binned", "ivar", "window", "nwin"]:
			data[field] = getattr(self, field)
		bunch.write(fname, data)
	@staticmethod
	def from_bunch(data):
		return NmatUncorr(spacing=data.spacing, nbin=data.nbin, nmin=data.nmin, bins=data.bins, ips_binned=data.ips_binned, ivar=data.ivar, window=window, nwin=nwin)

class NmatDetvecs(Nmat):
	def __init__(self, bin_edges=None, eig_lim=16, single_lim=0.55, mode_bins=[0.25,4.0,20],
			downweight=[], window=2, nwin=None, verbose=False, bins=None, D=None, V=None, iD=None, iV=None, s=None, ivar=None):
		# This is all taken from act, not tuned to so yet
		if bin_edges is None: bin_edges = np.array([
			0.16, 0.25, 0.35, 0.45, 0.55, 0.70, 0.85, 1.00,
			1.20, 1.40, 1.70, 2.00, 2.40, 2.80, 3.40, 3.80,
			4.60, 5.00, 5.50, 6.00, 6.50, 7.00, 8.00, 9.00, 10.0, 11.0,
			12.0, 13.0, 14.0, 16.0, 18.0, 20.0, 22.0,
			24.0, 26.0, 28.0, 30.0, 32.0, 36.5, 41.0,
			45.0, 50.0, 55.0, 65.0, 70.0, 80.0, 90.0,
			100., 110., 120., 130., 140., 150., 160., 170.,
			180., 190.
		])
		self.bin_edges = bin_edges
		self.mode_bins = mode_bins
		self.eig_lim   = np.zeros(len(mode_bins))+eig_lim
		self.single_lim= np.zeros(len(mode_bins))+single_lim
		self.verbose   = verbose
		self.downweight= downweight
		self.bins = bins
		self.window = window
		self.nwin   = nwin
		self.D, self.V, self.iD, self.iV, self.s, self.ivar = D, V, iD, iV, s, ivar
		self.ready      = all([a is not None for a in [D, V, iD, iV, s, ivar]])
	def build(self, tod, srate, extra=False, **kwargs):
		# Apply window before measuring noise model
		nwin  = utils.nint(self.window*srate)
		apply_window(tod, nwin)
		ft    = fft.rfft(tod)
		# Unapply window again
		apply_window(tod, nwin, -1)
		ndet, nfreq = ft.shape
		nsamp = tod.shape[1]
		# First build our set of eigenvectors in two bins. The first goes from
		# 0.25 to 4 Hz the second from 4Hz and up
		mode_bins = makebins(self.mode_bins, srate, nfreq, 1000, rfun=np.round)[1:]
		if np.any(np.diff(mode_bins) < 0):
			raise RuntimeError(f"At least one of the frequency bins has a negative range: \n{mode_bins}")
		# Then use these to get our set of basis vectors
		vecs = find_modes_jon(ft, mode_bins, eig_lim=self.eig_lim, single_lim=self.single_lim, verbose=self.verbose)
		nmode= vecs.shape[1]
		if vecs.size == 0: raise errors.ModelError("Could not find any noise modes")
		# Cut bins that extend beyond our max frequency
		bin_edges = self.bin_edges[self.bin_edges < srate/2 * 0.99]
		bins      = makebins(bin_edges, srate, nfreq, nmin=2*nmode, rfun=np.round)
		nbin      = len(bins)
		# Now measure the power of each basis vector in each bin. The residual
		# noise will be modeled as uncorrelated
		E  = np.zeros([nbin,nmode])
		D  = np.zeros([nbin,ndet])
		Nd = np.zeros([nbin,ndet])
		for bi, b in enumerate(bins):
			# Skip the DC mode, since it's it's unmeasurable and filtered away
			b = np.maximum(1,b)
			E[bi], D[bi], Nd[bi] = measure_detvecs(ft[:,b[0]:b[1]], vecs)
		# Optionally downweight the lowest frequency bins
		if self.downweight != None and len(self.downweight) > 0:
			D[:len(self.downweight)] /= np.array(self.downweight)[:,None]
		# Instead of VEV' we can have just VV' if we bake sqrt(E) into V
		V = vecs[None]*E[:,None]**0.5
		# At this point we have a model for the total noise covariance as
		# N = D + VV'. But since we're doing inverse covariance weighting
		# we need a similar representation for the inverse iN. The function
		# woodbury_invert computes iD, iV, s such that iN = iD + s iV iV'
		# where s usually is -1, but will become +1 if one inverts again
		iD, iV, s = woodbury_invert(D, V)
		# Also compute a representative white noise level
		bsize = bins[:,1]-bins[:,0]
		ivar  = np.sum(iD*bsize[:,None],0)/np.sum(bsize)
		# What about units? I haven't applied any fourier unit factors so far,
		# so we're in plain power units. From the uncorrelated model I found
		# that factor of tod.shape[1] is needed
		iD   *= nsamp
		iV   *= nsamp**0.5
		ivar *= nsamp
		# Fix dtype
		bins = np.ascontiguousarray(bins.astype(np.int32))
		D    = np.ascontiguousarray(D.astype(tod.dtype))
		V    = np.ascontiguousarray(V.astype(tod.dtype))
		iD   = np.ascontiguousarray(iD.astype(tod.dtype))
		iV   = np.ascontiguousarray(iV.astype(tod.dtype))
		nmat = NmatDetvecs(bin_edges=self.bin_edges, eig_lim=self.eig_lim, single_lim=self.single_lim,
				window=self.window, nwin=nwin, downweight=self.downweight, verbose=self.verbose,
				bins=bins, D=D, V=V, iD=iD, iV=iV, s=s, ivar=ivar)
		if extra: return nmat, bunch.Bunch(V=vecs, E=E, D=D)
		else:     return nmat
	def apply(self, tod, inplace=True, slow=False):
		if not inplace: tod = np.array(tod)
		apply_window(tod, self.nwin)
		ftod = fft.rfft(tod)
		norm = tod.shape[1]
		if slow:
			for bi, b in enumerate(self.bins):
				# Want to multiply by iD + siViV'
				ft    = ftod[:,b[0]:b[1]]
				iD    = self.iD[bi]/norm
				iV    = self.iV[bi]/norm**0.5
				ft[:] = iD[:,None]*ft + self.s*iV.dot(iV.T.dot(ft))
		else:
			so3g.nmat_detvecs_apply(ftod.view(tod.dtype), self.bins, self.iD, self.iV, float(self.s), float(norm))
		# I divided by the normalization above instead of passing normalize=True
		# here to reduce the number of operations needed
		fft.irfft(ftod, tod)
		apply_window(tod, self.nwin)
		return tod
	def white(self, tod, inplace=True):
		if not inplace: tod = np.array(tod)
		apply_window(tod, self.nwin)
		tod *= self.ivar[:,None]
		apply_window(tod, self.nwin)
		return tod
	def write(self, fname):
		data = bunch.Bunch(type="NmatDetvecs")
		for field in ["bin_edges", "eig_lim", "single_lim", "window", "nwin", "downweight",
				"bins", "D", "V", "iD", "iV", "s", "ivar"]:
			data[field] = getattr(self, field)
		bunch.write(fname, data)
	@staticmethod
	def from_bunch(data):
		return NmatDetvecs(bin_edges=data.bin_edges, eig_lim=data.eig_lim, single_lim=data.single_lim,
				window=data.window, nwin=data.nwin, downweight=data.downweight,
				bins=data.bins, D=data.D, V=data.V, iD=data.iD, iV=data.iV, s=data.s, ivar=data.ivar)

def measure_cov(d, nmax=10000):
	d = d[:,::max(1,d.shape[1]//nmax)]
	n,m = d.shape
	step  = 10000
	res = np.zeros((n,n),d.dtype)
	for i in range(0,m,step):
		sub = mycontiguous(d[:,i:i+step])
		res += np.real(sub.dot(np.conj(sub.T)))
	return res/m

def project_out(d, modes): return d-modes.T.dot(modes.dot(d))

def project_out_from_matrix(A, V):
	# Used Woodbury to project out the given vectors from the covmat A
	if V.size == 0: return A
	Q = A.dot(V)
	return A - Q.dot(np.linalg.solve(np.conj(V.T).dot(Q), np.conj(Q.T)))

def measure_power(d): return np.real(np.mean(d*np.conj(d),-1))

def freq2ind(freqs, srate, nfreq, rfun=None):
	"""Returns the index of the first fourier mode with greater than freq
	frequency, for each freq in freqs."""
	if freqs is None: return freqs
	if rfun  is None: rfun = np.ceil
	return rfun(np.asarray(freqs)/(srate/2.0)*nfreq).astype(int)

def makebins(edge_freqs, srate, nfreq, nmin=0, rfun=None):
	# Translate from frequency to index
	binds  = freq2ind(edge_freqs, srate, nfreq, rfun=rfun)
	# Make sure no bins have two few entries
	if nmin > 0:
		binds2 = [binds[0]]
		for b in binds:
			if b-binds2[-1] >= nmin: binds2.append(b)
		binds = binds2
	# Cap at nfreq and eliminate any resulting empty bins
	binds = np.unique(np.minimum(np.concatenate([[0],binds,[nfreq]]),nfreq))
	# Go from edges to [:,{from,to}]
	bins  = np.array([binds[:-1],binds[1:]]).T
	return bins

def mycontiguous(a):
	# I used this in act for some reason, but not sure why. I vaguely remember ascontiguousarray
	# causing weird failures later in lapack
	b = np.zeros(a.shape, a.dtype)
	b[...] = a[...]
	return b

def find_modes_jon(ft, bins, eig_lim=None, single_lim=0, skip_mean=False, verbose=False):
	ndet = ft.shape[0]
	vecs = np.zeros([ndet,0])
	if not skip_mean:
		# Force the uniform common mode to be included. This
		# assumes all the detectors have accurately measured gain.
		# Forcing this avoids the possibility that we don't find
		# any modes at all.
		vecs = np.concatenate([vecs,np.full([ndet,1],ndet**-0.5)],1)
	for bi, b in enumerate(bins):
		d    = ft[:,b[0]:b[1]]
		cov  = measure_cov(d)
		cov  = project_out_from_matrix(cov, vecs)
		e, v = np.linalg.eig(cov)
		e, v = e.real, v.real
		#e, v = e[::-1], v[:,::-1]
		accept = np.full(len(e), True, bool)
		if eig_lim is not None:
			# Compute median, exempting modes we don't have enough data to measure
			nsamp    = b[1]-b[0]+1
			median_e = np.median(np.sort(e)[::-1][:nsamp])
			accept  &= e/median_e >= eig_lim[bi]
		if verbose: print("bin %d: %4d modes above eig_lim" % (bi, np.sum(accept)))
		if single_lim is not None and e.size:
			# Reject modes too concentrated into a single mode. Since v is normalized,
			# values close to 1 in a single component must mean that all other components are small
			singleness = np.max(np.abs(v),0)
			accept    &= singleness < single_lim[bi]
		if verbose: print("bin %d: %4d modes also above single_lim" % (bi, np.sum(accept)))
		e, v = e[accept], v[:,accept]
		vecs = np.concatenate([vecs,v],1)
	return vecs

def measure_detvecs(ft, vecs):
	# Measure amps when we have non-orthogonal vecs
	rhs  = vecs.T.dot(ft)
	div  = vecs.T.dot(vecs)
	amps = np.linalg.solve(div,rhs)
	E    = np.mean(np.abs(amps)**2,1)
	# Project out modes for every frequency individually
	dclean = ft - vecs.dot(amps)
	# The rest is assumed to be uncorrelated
	Nu = np.mean(np.abs(dclean)**2,1)
	# The total auto-power
	Nd = np.mean(np.abs(ft)**2,1)
	return E, Nu, Nd

def sichol(A):
	iA = np.linalg.inv(A)
	try: return np.linalg.cholesky(iA), 1
	except np.linalg.LinAlgError:
		return np.linalg.cholesky(-iA), -1

def safe_inv(a):
	with utils.nowarn():
		res = 1/a
		res[~np.isfinite(res)] = 0
	return res

def woodbury_invert(D, V, s=1):
	"""Given a compressed representation C = D + sVV', compute a
	corresponding representation for inv(C) using the Woodbury
	formula."""
	V, D = map(np.asarray, [V,D])
	# Flatten everything so we can be dimensionality-agnostic
	D = D.reshape(-1, D.shape[-1])
	V = V.reshape(-1, V.shape[-2], V.shape[-1])
	I = np.eye(V.shape[2])
	# Allocate our output arrays
	iD = safe_inv(D)
	iV = V*0
	# Invert each
	for i in range(len(D)):
		core = I*s + (V[i].T*iD[i,None,:]).dot(V[i])
		core, sout = sichol(core)
		iV[i] = iD[i,:,None]*V[i].dot(core)
	sout = -sout
	return iD, iV, sout

def apply_window(tod, nsamp, exp=1):
	"""Apply a cosine taper to each end of the TOD."""
	if nsamp <= 0: return
	taper   = 0.5*(1-np.cos(np.arange(1,nsamp+1)*np.pi/nsamp))
	taper **= exp
	tod[...,:nsamp]  *= taper
	tod[...,-nsamp:] *= taper[::-1]

# Signal classes represent the degrees of freedom we will solve for.
# The Zippers should probably be merged with these

class Signal:
	"""This class represents a thing we want to solve for, e.g. the sky, ground, cut samples, etc."""
	def __init__(self, name, ofmt, output, ext):
		"""Initialize a Signal. It probably doesn't make sense to construct a generic signal
		directly, though. Use one of the subclasses.
		Arguments:
		* name: The name of this signal, e.g. "sky", "cut", etc.
		* ofmt: The format used when constructing output file prefix
		* output: Whether this signal should be part of the output or not.
		* ext: The extension used for the files.
		"""
		self.name   = name
		self.ofmt   = ofmt
		self.output = output
		self.ext    = ext
		self.dof    = None
		self.ready  = False
	def add_obs(self, id, obs, nmat, Nd): pass
	def prepare(self): self.ready = True
	def forward (self, id, tod, x): pass
	def backward(self, id, tod, x): pass
	def precon(self, x): return x
	def to_work  (self, x): return x.copy()
	def from_work(self, x): return x
	def write   (self, prefix, tag, x): pass

class SignalMap(Signal):
	"""Signal describing a non-distributed sky map."""
	def __init__(self, shape, wcs, comm, name="sky", ofmt="{name}", output=True,
			ext="fits", dtype=np.float32, sys=None, interpol=None):
		"""Signal describing a sky map in the coordinate system given by "sys", which defaults
		to equatorial coordinates. If tiled==True, then this will be a distributed map with
		the given tile_shape, otherwise it will be a plain enmap. interpol controls the
		pointing matrix interpolation mode. See so3g's Projectionist docstring for details."""
		Signal.__init__(self, name, ofmt, output, ext)
		self.comm  = comm
		self.sys   = sys
		self.dtype = dtype
		self.interpol = interpol
		self.data  = {}
		self.comps = "TQU"
		self.ncomp = 3
		shape      = tuple(shape[-2:])
		self.rhs = enmap.zeros((self.ncomp,)+shape, wcs, dtype=dtype)
		self.div = enmap.zeros(              shape, wcs, dtype=dtype)
		self.hits= enmap.zeros(              shape, wcs, dtype=dtype)
	def add_obs(self, id, obs, nmat, Nd, pmap=None):
		"""Add and process an observation, building the pointing matrix
		and our part of the RHS. "obs" should be an Observation axis manager,
		nmat a noise model, representing the inverse noise covariance matrix,
		and Nd the result of applying the noise model to the detector time-ordered data.
		"""
		Nd	 = Nd.copy() # This copy can be avoided if build_obs is split into two parts
		ctime  = obs.ctime
		t1     = time.time()
		print("H", so3g.useful_info()["omp_num_threads"])
		pcut   = PmatCut(obs.cuts) # could pass this in, but fast to construct
		if pmap is None:
			# Build the local geometry and pointing matrix for this observation
			focal_plane = bunch.Bunch(eta=obs.point_offset[:,0], xi=obs.point_offset[:,1], gamma=np.pi/2-obs.polangle)
			# Hack: make fp.get("gamma") work
			focal_plane.get = lambda name: focal_plane.gamma
			print("I", so3g.useful_info()["omp_num_threads"])
			pmap = coords.pmat.P.for_tod(
				obs,
				timestamps  = obs.ctime,
				boresight   = bunch.Bunch(el =obs.boresight[0], az=obs.boresight[1], roll=None),
				focal_plane = focal_plane,
				comps=self.comps, geom=self.rhs.geometry,
				threads="domdir", weather="typical", site="so", interpol=self.interpol)
		print("J", so3g.useful_info()["omp_num_threads"])
		# Build the RHS for this observation
		t2 = time.time()
		pcut.clear(Nd)
		obs_rhs = pmap.zeros()
		pmap.to_map(dest=obs_rhs, signal=Nd)
		t3 = time.time()
		# Build the per-pixel inverse variance for this observation.
		# This will be scalar to make the preconditioner fast, but uses
		# ncomp while building since pmat expects that
		Nd[:]        = 1
		pcut.clear(Nd)
		Nd = nmat.white(Nd)
		obs_div = pmap.to_map(signal=Nd)[0]
		t4 = time.time()
		# Build hitcount
		Nd[:] = 1
		pcut.clear(Nd)
		obs_hits = pmap.to_map(signal=Nd)[0]
		t5 = time.time()
		del Nd
		# Update our full rhs and div. This works for both plain and distributed maps
		self.rhs = self.rhs .insert(obs_rhs, op=np.ndarray.__iadd__)
		self.div = self.div .insert(obs_div, op=np.ndarray.__iadd__)
		self.hits= self.hits.insert(obs_hits,op=np.ndarray.__iadd__)
		t6 = time.time()
		L.print("Init map pmat %6.3f rhs %6.3f div %6.3f hit %6.3f add %6.3f %s" % (t2-t1,t3-t2,t4-t3,t5-t4,t6-t5,id), level=2)
		# Save the per-obs things we need. Just the pointing matrix in our case.
		# Nmat and other non-Signal-specific things are handled in the mapmaker itself.
		self.data[id] = bunch.Bunch(pmap=pmap, obs_geo=obs_rhs.geometry)
	def prepare(self):
		"""Called when we're done adding everything. Sets up the map distribution,
		degrees of freedom and preconditioner."""
		if self.ready: return
		t1 = time.time()
		if self.comm is not None:
			self.rhs  = utils.allreduce(self.rhs, self.comm)
			self.div  = utils.allreduce(self.div, self.comm)
			self.hits = utils.allreduce(self.hits,self.comm)
		self.dof   = MapZipper(*self.rhs.geometry, dtype=self.dtype)
		self.idiv  = safe_inv(self.div)
		t2 = time.time()
		L.print("Prep map %6.3f" % (t2-t1), level=2)
		self.ready = True
	def forward(self, id, tod, map, tmul=1, mmul=1):
		"""map2tod operation. For tiled maps, the map should be in work distribution,
		as returned by unzip. Adds into tod."""
		if id not in self.data: return # Should this really skip silently like this?
		if tmul != 1: tod *= tmul
		if mmul != 1: map = map*mmul
		self.data[id].pmap.from_map(dest=tod, signal_map=map, comps=self.comps)
	def backward(self, id, tod, map, tmul=1, mmul=1):
		"""tod2map operation. For tiled maps, the map should be in work distribution,
		as returned by unzip. Adds into map"""
		if id not in self.data: return
		if tmul != 1: tod  = tod*tmul
		if mmul != 1: map *= mmul
		self.data[id].pmap.to_map(signal=tod, dest=map, comps=self.comps)
	def precon(self, map):
		return self.idiv * map
	def to_work(self, map):
		return map.copy()
	def from_work(self, map):
		if self.comm is None: return map
		else: return utils.allreduce(map, self.comm)
	def write(self, prefix, tag, m):
		if not self.output: return
		oname = self.ofmt.format(name=self.name)
		oname = "%s%s_%s.%s" % (prefix, oname, tag, self.ext)
		if self.comm is None or self.comm.rank == 0:
			enmap.write_map(oname, m)
		return oname

class SignalCut(Signal):
	def __init__(self, comm, name="cut", ofmt="{name}_{rank:02}", dtype=np.float32,
			output=False, cut_type=None):
		"""Signal for handling the ML solution for the values of the cut samples."""
		Signal.__init__(self, name, ofmt, output, ext="hdf")
		self.comm  = comm
		self.data  = {}
		self.dtype = dtype
		self.cut_type = cut_type
		self.off   = 0
		self.rhs   = []
		self.div   = []
	def add_obs(self, id, obs, nmat, Nd):
		"""Add and process an observation. "obs" should be an Observation axis manager,
		nmat a noise model, representing the inverse noise covariance matrix,
		and Nd the result of applying the noise model to the detector time-ordered data."""
		Nd      = Nd.copy() # This copy can be avoided if build_obs is split into two parts
		pcut    = PmatCut(obs.cuts, model=self.cut_type)
		# Build our RHS
		obs_rhs = np.zeros(pcut.njunk, self.dtype)
		pcut.backward(Nd, obs_rhs)
		# Build our per-pixel inverse covmat
		obs_div = np.ones(pcut.njunk, self.dtype)
		Nd[:]   = 0
		pcut.forward(Nd, obs_div)
		Nd     *= nmat.ivar[:,None]
		pcut.backward(Nd, obs_div)
		self.data[id] = bunch.Bunch(pcut=pcut, i1=self.off, i2=self.off+pcut.njunk)
		self.off += pcut.njunk
		self.rhs.append(obs_rhs)
		self.div.append(obs_div)
	def prepare(self):
		"""Process the added observations, determining our degrees of freedom etc.
		Should be done before calling forward and backward."""
		if self.ready: return
		self.rhs = np.concatenate(self.rhs)
		self.div = np.concatenate(self.div)
		self.dof = ArrayZipper(self.rhs.shape, dtype=self.dtype, comm=self.comm)
		self.ready = True
	def forward(self, id, tod, junk):
		if id not in self.data: return
		d = self.data[id]
		d.pcut.forward(tod, junk[d.i1:d.i2])
	def precon(self, junk):
		return junk/self.div
	def backward(self, id, tod, junk):
		if id not in self.data: return
		d = self.data[id]
		d.pcut.backward(tod, junk[d.i1:d.i2])
	def write(self, prefix, tag, m):
		if not self.output: return
		if self.comm is None:
			rank = 0
		else:
			rank = self.comm.rank
		oname = self.ofmt.format(name=self.name, rank=rank)
		oname = "%s%s_%s.%s" % (prefix, oname, tag, self.ext)
		with h5py.File(oname, "w") as hfile:
			hfile["data"] = m
		return oname


class MLMapmaker:
	def __init__(self, signals=[], noise_model=None, dtype=np.float32, verbose=False):
		"""Initialize a Maximum Likelihood Mapmaker.
		Arguments:
		* signals: List of Signal-objects representing the models that will be solved
		  jointly for. Typically this would be the sky map and the cut samples. NB!
		  The way the cuts currently work, they *MUST* be the first signal specified.
		  If not, the equation system will be inconsistent and won't converge.
		* noise_model: A noise model constructor which will be used to initialize the
		  noise model for each observation. Can be overriden in add_obs.
		* dtype: The data type to use for the time-ordered data. Only tested with float32
		* verbose: Whether to print progress messages. Not implemented"""
		self.signals  = signals
		self.dtype    = dtype
		self.verbose  = verbose
		self.noise_model = noise_model or NmatUncorr()
		self.data     = []
		self.dof      = MultiZipper()
		self.ready    = False
	def add_obs(self, id, obs, deslope=True, noise_model=None):
		# Prepare our tod
		t1 = time.time()
		ctime  = obs.ctime
		srate  = (len(ctime)-1)/(ctime[-1]-ctime[0])
		tod    = obs.tod.astype(self.dtype, copy=False)
		t2 = time.time()
		if deslope:
			utils.deslope(tod, w=5, inplace=True)
		t3 = time.time()
		print("E", so3g.useful_info()["omp_num_threads"])
		# Allow the user to override the noise model on a per-obs level
		if noise_model is None: noise_model = self.noise_model
		# Build the noise model from the obs unless a fully
		# initialized noise model was passed
		if noise_model.ready:
			nmat = noise_model
		else:
			try:
				nmat = noise_model.build(tod, srate=srate)
			except Exception as e:
				msg = f"FAILED to build a noise model for observation='{id}' : '{e}'"
				raise RuntimeError(msg)
		print("F", so3g.useful_info()["omp_num_threads"])
		t4 = time.time()
		# And apply it to the tod
		tod = nmat.apply(tod)
		t5 = time.time()
		print("G", so3g.useful_info()["omp_num_threads"])
		# Add the observation to each of our signals
		for signal in self.signals:
			signal.add_obs(id, obs, nmat, tod)
		t6 = time.time()
		L.print("Init sys trun %6.3f ds %6.3f Nb %6.3f N %6.3f add sigs %6.3f %s" % (t2-t1, t3-t2, t4-t3, t5-t4, t6-t5, id), level=2)
		# Save only what we need about this observation
		self.data.append(bunch.Bunch(id=id, ndet=len(obs.dets), nsamp=len(ctime),
			dets=obs.dets, nmat=nmat))
	def prepare(self):
		if self.ready: return
		t1 = time.time()
		for signal in self.signals:
			signal.prepare()
			self.dof.add(signal.dof)
		t2 = time.time()
		L.print("Prep sys %6.3f" % (t2-t1), level=2)
		self.ready = True
	def A(self, x):
		t1 = time.time()
		iwork = [signal.to_work(m) for signal,m in zip(self.signals,self.dof.unzip(x))]
		owork = [w*0 for w in iwork]
		t2 = time.time()
		for di, data in enumerate(self.data):
			# This is the main place that needs to change for the GPU implementation
			ta1 = time.time()
			tod = np.zeros([data.ndet, data.nsamp], self.dtype)
			ta2 = time.time()
			for si, signal in reversed(list(enumerate(self.signals))):
				signal.forward(data.id, tod, iwork[si])
			ta3 = time.time()
			data.nmat.apply(tod)
			ta4 = time.time()
			for si, signal in enumerate(self.signals):
				signal.backward(data.id, tod, owork[si])
			ta5 = time.time()
			L.print("A z %6.3f P %6.3f N %6.3f P' %6.3f %s" % (ta2-ta1, ta3-ta2, ta4-ta3, ta5-ta4, data.id), level=2)
		t3 = time.time()
		result = self.dof.zip(*[signal.from_work(w) for signal,w in zip(self.signals,owork)])
		t4 = time.time()
		L.print("A prep %6.3f PNP %6.3f finish %6.3f" % (t2-t1, t3-t2, t4-t3), level=2)
		return result
	def M(self, x):
		t1 = time.time()
		iwork = self.dof.unzip(x)
		result = self.dof.zip(*[signal.precon(w) for signal, w in zip(self.signals, iwork)])
		t2 = time.time()
		L.print("M %6.3f" % (t2-t1), level=2)
		return result
	def solve(self, maxiter=500, maxerr=1e-6):
		self.prepare()
		rhs    = self.dof.zip(*[signal.rhs for signal in self.signals])
		solver = utils.CG(self.A, rhs, M=self.M, dot=self.dof.dot)
		while solver.i < maxiter and solver.err > maxerr:
			t1 = time.time()
			solver.step()
			x  = self.dof.unzip(solver.x)
			t2 = time.time()
			yield bunch.Bunch(i=solver.i, err=solver.err, x=x, t=t2-t1)

if __name__ == "__main__":
	print("A", so3g.useful_info()["omp_num_threads"])

	ifiles = sum([sorted(utils.glob(ifile)) for ifile in args.ifiles],[])
	nfile  = len(ifiles)
	comm   = mpi.COMM_WORLD
	dtype_tod = np.float32
	dtype_map = np.float64
	shape, wcs = enmap.read_map_geometry(args.area)
	L      = Logger(id=comm.rank, level=args.verbose-args.quiet)
	L.print("Mapping %d tods with %d mpi tasks" % (nfile, comm.size), level=0, id=0, color=colors.lgreen)
	prefix = args.odir + "/"
	if args.prefix: prefix += args.prefix + "_"
	utils.mkdir(args.odir)
	# Set up the signals we will solve for
	signal_map = SignalMap(shape, wcs, comm, dtype=dtype_map)
	signal_cut = SignalCut(comm)
	print("B", so3g.useful_info()["omp_num_threads"])
	# Set up the mapmaker
	mapmaker = MLMapmaker(signals=[signal_cut,signal_map], dtype=dtype_tod, verbose=True, noise_model=NmatDetvecs())
	print("C", so3g.useful_info()["omp_num_threads"])
	# Add our observations
	for ind in range(comm.rank, nfile, comm.size):
		ifile = ifiles[ind]
		id    = ".".join(os.path.basename(ifile).split(".")[:-1])
		t1    = time.time()
		data  = read_tod(ifile)
		t2    = time.time()
		print("D", so3g.useful_info()["omp_num_threads"])
		mapmaker.add_obs(id, data, deslope=False)
		t3    = time.time()
		L.print("Processed %s in %6.3f. Read %6.3f Add %6.3f" % (id, t3-t1, t2-t1, t3-t2))
	# Solve the equation system
	for step in mapmaker.solve():
		L.print("CG %4d %15.7e (%6.3f s)" % (step.i, step.err, step.t), id=0, level=1, color=colors.lgreen)
		if step.i % 10 == 0:
			for signal, val in zip(mapmaker.signals, step.x):
				if signal.output:
					signal.write(prefix, "map%04d" % step.i, val)
