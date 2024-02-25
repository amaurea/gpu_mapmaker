import numpy as np, time
import cupy as cp
import time
import glob
import ctypes
import gpu_mm

# Cuts will be represented by det[nrange], start[nrange], len[nrange]. This is similar to
# the format used in ACT, but in our test files we have boolean masks instead, which we need
# convert. This is a bit slow, but is only needed for the test data
def mask_to_ranges(mask):
	# Find where the mask turns on/off
	t01 = time.time()
	dets, starts, lens = [], [], []
	for idet, dmask in enumerate(mask):
		# index of all on/off and off/on transitions. We put it in a
		# list so we can prepend and append to it
		edges = [1+np.nonzero(np.diff(dmask,1))[0]]
		# Ensure we start with off→on and end with on→off
		if dmask[ 0]: edges.insert(0,[0])
		if dmask[-1]: edges.append([mask.shape[1]])
		edges = np.concatenate(edges) if len(edges) > 1 else edges[0]
		start = edges[0::2].astype(np.int32)
		stop  = edges[1::2].astype(np.int32)
		dets  .append(np.full(len(start),idet,np.int32))
		starts.append(start)
		lens  .append(stop-start)
	dets   = np.concatenate(dets)
	starts = np.concatenate(starts)
	lens   = np.concatenate(lens)
	t02 = time.time()
	return dets, starts, lens

def aranges(lens):
	ntot = np.sum(lens)
	itot = np.arange(ntot)
	offs = np.repeat(np.cumsum(np.concatenate([[0],lens[:-1]])), lens)
	return itot-offs

def cumsum0(vals):
	res = np.empty(len(vals), vals.dtype)
	res[0] = 0
	res[1:] = np.cumsum(vals[:-1])
	return res

def split_ranges(dets, starts, lens, maxlen):
	# Vectorized splitting of detector ranges into subranges.
	# Probably premature optimization, since it's a bit hard to read.
	# Works by duplicating elements for too long ranges, and then
	# calculating new sub-offsets inside these
	dets, starts, lens = [np.asarray(a) for a in [dets,starts,lens]]
	nsplit  = (lens+maxlen-1)//maxlen
	odets   = np.repeat(dets, nsplit)
	subi    = aranges(nsplit)
	sublen  = np.repeat(lens, nsplit)
	subns   = np.repeat(nsplit, nsplit)
	offs    = subi*sublen//subns
	ostarts = np.repeat(starts, nsplit)+offs
	offs2   = (subi+1)*sublen//subns
	oends   = offs2-offs
	return odets, ostarts, oends

def legbasis_gpu(order, n):
	x   = cp.linspace(-1, 1, n, dtype=np.float32)
	out = cp.empty((order+1, n),dtype=np.float32)
	out[0] = 1
	if order>0:
		out[1] = x
	for i in range(1,order):
		out[i+1,:] = ((2*i+1)*x*out[i]-i*out[i-1])/(i+1)
	return out

class PmatCutsFull:
	def __init__(self, dets, starts, lens):
		self.dets   = cp.asarray(dets,   np.int32)
		self.starts = cp.asarray(starts, np.int32)
		self.lens   = cp.asarray(lens,   np.int32)
		self.ndof   = np.sum(lens)  # number of values to solve for
		self.nsamp  = self.ndof     # number of samples covered
		self.offs   = cp.asarray(cumsum0(lens), np.int32)
	def forward(self, tod, junk):
		gpu_mm.insert_ranges(tod, junk, self.offs, self.dets, self.starts, self.lens)
	def backward(self, tod, junk):
		gpu_mm.extract_ranges(tod, junk, self.offs, self.dets, self.starts, self.lens)
		# Zero-out the samples we used, so the other signals (e.g. the map)
		# don't need to care about them
		gpu_mm.clear_ranges(tod, self.dets, self.starts, self.lens)

class PmatCutsPoly:
	def __init__(self, dets, starts, lens, basis=None, order=None, bsize=None):
		# Either construct or use an existing basis
		if basis is None:
			if bsize is None: bsize = 400
			if order is None: order = 4
			self.basis = legbasis_gpu(order, bsize)
		else:
			assert order is None and bsize is None, "Specify either basis or order,bsize, not both"
			order = basis.shape[0]-1
			bsize = basis.shape[1]
			self.basis = cp.asarray(basis, dtype=np.float32)
		# Subdivide ranges that are longer than our block size
		dets, starts, lens = split_ranges(dets, starts, lens, bsize)
		self.dets   = cp.asarray(dets,   np.int32)
		self.starts = cp.asarray(starts, np.int32)
		self.lens   = cp.asarray(lens,   np.int32)
		# total number of samples covered
		self.nsamp  = np.sum(lens)
		# output buffer information. Offsets
		padlens     = (lens+bsize-1)//bsize*bsize
		self.nrange = len(lens)
		self.ndof   = self.nrange*(order+1)
		self.offs   = cp.asarray(cumsum0(padlens), np.int32)
	def forward(self, tod, junk):
		# B[nb,bsize], bjunk[nrange,nb], blocks[nrange,bsize] = bjunk.dot(B.T)
		bjunk  = junk.reshape(self.nrange,self.basis.shape[0])
		# TODO: avoid allocation here by using a gpu buffer
		blocks = bjunk.dot(self.basis)
		gpu_mm.insert_ranges(tod, blocks, self.offs, self.dets, self.starts, self.lens)
	def backward(self, tod, junk):
		# TODO: allocate bdata using a gpu buffer here
		blocks = cp.empty((self.nrange, self.basis.shape[1]), np.float32)
		gpu_mm.extract_ranges(tod, blocks, self.offs, self.dets, self.starts, self.lens)
		gpu_mm.clear_ranges(tod, self.dets, self.starts, self.lens)
		bjunk  = blocks.dot(self.basis.T)
		junk[:] = bjunk.reshape(-1)

dr='/home/sigurdkn/gpu_mapmaker/tods/'
fnames=glob.glob(dr+'tod*.npz')

np.random.seed(1)

#read a random sampling of TODs and set up their cuts structures
try:
	print('cuts shape is ',len(cuts))
except:
	ords=np.random.permutation(np.arange(len(fnames)))
	n_to_test=10
	fnames=[fnames[i] for i in ords[:n_to_test]]

	cuts=[None]*n_to_test
	n=[None]*n_to_test
	ndet=[None]*n_to_test
	basis = legbasis_gpu(4, 400)
	for i in range(len(fnames)):
		t1 = time.time()
		tod = np.load(fnames[i])
		#key line to get the cuts, do not forget the logical not!
		mask = tod["cuts"]
		t2 = time.time()
		dets, starts, lens = mask_to_ranges(mask)
		t3 = time.time()
		cuts[i] = PmatCutsPoly(dets, starts, lens, basis=basis)
		t4 = time.time()
		ndet[i], n[i] = mask.shape
		t5 = time.time()
		print("%8.3f %8.3f %8.3f %8.3f %s" % (1e3*(t2-t1), 1e3*(t3-t2), 1e3*(t4-t3), 1e3*(t5-t4), fnames[i]))
	nmax = np.max([n[i]*ndet[i] for i in range(n_to_test)])

#set up some random data so we can make sure the cuts did what they should have
try:
	print('big tod shape is ',big_tod.shape)
except:	
	big_tod=cp.random.randn(nmax).astype('float32')
	tods=[None]*n_to_test
	for i in range(n_to_test):
		tods[i]=cp.reshape(big_tod[:n[i]*ndet[i]],[ndet[i],n[i]])
	junks=[cp.zeros(cut.ndof, np.float32) for cut in cuts]

#loop over tod2map and map2tod, with timing
niter=10
for ii in range(niter):
	cp.cuda.runtime.deviceSynchronize()
	t1=time.time()
	for i in range(n_to_test):
		cuts[i].forward (tods[i], junks[i])
		cuts[i].backward(tods[i], junks[i])
	cp.cuda.runtime.deviceSynchronize()
	t2=time.time()
	print('average time per TOD was %8.5f ms' % ((t2-t1)/n_to_test*1e3))

# This test is meaningless for the poly case
tods[i][:]=0
cuts[i].backward(tods[i], junks[i])
junks[i][:]=1
cuts[i].forward (tods[i], junks[i])
print('sum is ',cp.sum(tods[i]))
print('expected ',cp.sum(cuts[i].nsamp))
