import numpy as np, so3g

def calc_pointing(ctime, bore, offs, polang, bsize=200, dtype=np.float32):
	ndet, nsamp = len(offs), bore.shape[1]
	sightline = so3g.proj.coords.CelestialSightLine.az_el(ctime, bore[1], bore[0], site="so", weather="typical")
	q_det     = so3g.proj.quat.rotation_xieta(offs[:,1], offs[:,0], np.pi/2-polang)
	pos_equ   = np.zeros((4,ndet,nsamp),dtype) # [{ra,dec,c,s},ndet,nsamp]
	for d1 in range(0, ndet, bsize):
		d2  = min(d1+bsize, ndet)
		res = np.asarray(sightline.coords(q_det[d1:d2]))
		pos_equ[1::-1,d1:d2] = np.moveaxis(res,2,0)[:2]
		pos_equ[2,d1:d2], pos_equ[3,d1:d2] = double_angle(res[...,2],res[...,3])
		del res
	return pos_equ

def double_angle(cos, sin):
	sin2 = 2*cos*sin
	cos2 = cos**2-sin**2
	return cos2, sin2

if __name__ == "__main__":
	import argparse, time
	parser = argparse.ArgumentParser()
	parser.add_argument("ifile")
	parser.add_argument("-d", "--dets", type=str, default=":1")
	parser.add_argument("-s", "--samps",type=str, default=":30")
	args = parser.parse_args()
	from pixell import utils

	f      = np.load(args.ifile)
	bore   = f["boresight"]
	offs   = f["point_offset"]
	polang = f["polangle"]
	if args.dets:
		offs   = eval("offs[%s]"   % args.dets)
		polang = eval("polang[%s]" % args.dets)
	if args.samps:
		bore   = eval("bore[:,%s]" % args.samps)
	pos_equ = calc_pointing(bore[0], bore[1:][::-1], offs[::-1], polang)
	ndet, nsamp = pos_equ.shape[1:]
	for i in range(nsamp):
		msg = "%8.3f %8.4f %7.4f" % (bore[0,i]-bore[0,0], bore[2,i]/utils.degree, bore[1,i]/utils.degree)
		for j in range(ndet):
				msg += " %8.4f" % (pos_equ[1,j,i]/utils.degree)
				msg += " %8.4f" % (pos_equ[0,j,i]/utils.degree)
				msg += " %6.3f" % (pos_equ[2,j,i])
				msg += " %6.3f" % (pos_equ[3,j,i])
		print(msg)
