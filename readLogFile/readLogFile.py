import numpy as np
from struct import unpack
class ReadLogFile(object):
	""" Reading sonar logfiles
	"""
	#F
	## Fileinfo
	header =""
	version=0
	regOffset=0
	dataOffset=0
	nScanLines=0
	configOffset=0
	extraOffset=0
	indexOffset=0
	checkOffset=0
	#tOpen
	#tClose
	scanLines = 0
	
	def __init__(self, filename):
		binary_file = open(filename, "rb")
		data = binary_file.read(80)
		# Log header information
		(self.header, self.version, self.regOffset, self.dataOffset, self.nScanLines, self.configOffset, 
		self.extraOffset, self.indexOffset, self.checkOffset, self.tOpen, self.tClose) = unpack('<32sIIIIIIIIdd', data)
		
		
		#scanLines
		binary_file.seek(self.dataOffset)
		#self.nScanLines = 1 #################################HUSK Å FJERNE"""""#################
		self.scanLines = np.zeros(self.nScanLines, dtype= [('length', np.uint16),
															('time', float),
															('tNode', np.uint8),
															('rNode', np.uint8),
															('msgType', np.uint8),
															('packetSeq', np.uint8),
															('nodeN', np.uint8),
															('length2', np.uint16),
															('headType', np.uint8),
															('status', np.uint8),
															('sweepCode', np.uint8),
															('headControls', np.uint16),
															('rangeScale', np.uint16),
															('transParam', np.uint64),
															('gain', np.uint8),
															('slope', np.uint16),
															('adSpan', np.uint8),
															('adLow', np.uint8),
															('headingOffset', np.uint16),
															('adInterval', np.uint16),
															('leftLim', np.uint16),
															('rightLim', np.uint16),
															('motorStep', np.uint8),
															('bearing', np.uint16),
															('scanLineBytes', np.uint16),
															('data', tuple)])
		n = 0
		for i in range(0, self.nScanLines):
			data = binary_file.read(46)
			(self.scanLines[n]['length'], self.scanLines[n]['time'], self.scanLines[n]['tNode'], self.scanLines[n]['rNode'], 
			self.scanLines[n]['msgType'], self.scanLines[n]['packetSeq'], self.scanLines[n]['nodeN'], self.scanLines[n]['length2'], 
			self.scanLines[n]['headType'], self.scanLines[n]['status'], self.scanLines[n]['sweepCode'], self.scanLines[n]['headControls'], 
			self.scanLines[n]['rangeScale'], self.scanLines[n]['transParam'], self.scanLines[n]['gain'], self.scanLines[n]['slope'], 
			self.scanLines[n]['adSpan'], self.scanLines[n]['adLow'], self.scanLines[n]['headingOffset'], self.scanLines[n]['adInterval'], 
			self.scanLines[n]['leftLim'], self.scanLines[n]['rightLim'], self.scanLines[n]['motorStep'], self.scanLines[n]['bearing'], 
			self.scanLines[n]['scanLineBytes']) = unpack('<HdBBBBBHBBBHHLBHBBHHHHBHH', data)
			
			if self.scanLines[n]['msgType'] == 2:
				data = binary_file.read(self.scanLines[n]['scanLineBytes'])
				adc8On = (self.scanLines[n]['headControls'] & 1 != 0) 
				if adc8On:
					self.scanLines[n]['data'] = unpack(('<'+str(self.scanLines[n]['scanLineBytes'])+'B'), data)
				else:
					raise NotImplementedError("adc8off not yet implemented")
				n=n+1
			else:
				binary_file.seek(self.scanLines[n]['scanLineBytes'], 1)
		binary_file.close()
		print(n)
			
"""
		%Scanlines
		fseek(fId, Log.Header.DataOffset, 'bof');
		j = 1;
		for i = 1:Log.Header.nScanLines
			Log.ScanLines.Position(j) = uint64(ftell(fId));
			Log.ScanLines.Length(j) = fread(fId, 1, '*uint16'); %1,2
			Log.ScanLines.Time(j) = fread(fId, 1, 'double'); %3-10
			fseek(fId, 2, 'cof');
			Log.ScanLines.Msg_type(j) = fread(fId, 1, '*uint8'); %13
			if Log.ScanLines.Msg_type(j) ~=2
				Log.ScanLines.IsScanLine(j) = false;
				fseek(fId, Log.ScanLines.Length(j)-13, 'cof');
			else
				Log.ScanLines.IsScanLine(j) = true;
				fseek(fId, Log.ScanLines.Length(j)-13, 'cof');
				j = j+1;
			end
		end
		fclose(fId);
		"""
