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
	#scanLines
	
	def __init__(self, filename):
		with open(filename, "rb") as binary_file:
			data = binary_file.read(80)
		# Log header information
		#self.header = 		str((unpack('s', data[0:32]))[0])
		#self.version = 		int((unpack('I', data[32:36]))[0])
		#self.regOffset = 	int((unpack('I', data[36:40]))[0])
		#self.dataOffset = 	int((unpack('I', data[40:44]))[0])
		#self.nScanLines = 	int((unpack('I', data[44:48]))[0])
		#self.configOffset = int((unpack('I', data[48:52]))[0])
		#self.extraOffset = 	int((unpack('I', data[52:56]))[0])
		#self.indexOffset = 	int((unpack('I', data[56:60]))[0])
		#self.checkOffset = 	int((unpack('I', data[60:64]))[0])
		#self.tOpen = 		float((unpack('d', data[64:72]))[0])
		#self.tClose = 		float((unpack('d', data[72:80]))[0])
		self.header, self.version, self.regOffset, self.dataOffset, self.nScanLines, self.configOffset, self.extraOffset, self.indexOffset, self.checkOffset, self.tOpen, self.tClose = unpack('<32sIIIIIIIIdd', data);
		
		binary_file.close()
			#scanLines
		
		
		"""
		Log.Header.Header = fscanf(fId, '%32c', 1);
		Log.Header.Version = fread(fId, 1, '*uint32');
		Log.Header.RegOffset = fread(fId, 1, '*uint32');
		Log.Header.DataOffset = fread(fId, 1, '*uint32');
		Log.Header.nScanLines = fread(fId, 1, '*uint32');
		Log.Header.ConfigOffset = fread(fId, 1, '*uint32');
		Log.Header.ExtraOffset = fread(fId, 1, '*uint32');
		Log.Header.IndexOffset = fread(fId, 1, '*uint32');
		Log.Header.CheckOffset = fread(fId, 1, '*uint32');
		Log.Header.LogOpen = datetime(1899, 12, 30) + days(fread(fId, 1, 'double'));
		Log.Header.LogClose = datetime(1899, 12, 30) + days(fread(fId, 1, 'double'));

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
