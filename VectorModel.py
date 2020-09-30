import json
import math
import time
import pickle

def calDistance(vec1, vec2):
	if len(vec1) != len(vec2):
		print("vector length not equal")
	else:
		tmp1 = 0
		tmp2 = 0
		tmp3 = 0
		for i in range(len(vec1)):
			tmp1 += vec1[i] * vec2[i]
			tmp2 += vec1[i] * vec1[i]
			tmp3 += vec2[i] * vec2[i]
		tmp2 = math.sqrt(tmp2)
		tmp3 = math.sqrt(tmp3)
		return float(tmp1 / (tmp2 * tmp3))

def quickSort(uid_list, dis_list, left, right):
		l = left
		r = right
		if l >= r:
			return
		key = dis_list[l]
		tmp = uid_list[l]
		while l < r:
			while dis_list[r] >= key and l < r:
				r -= 1
			dis_list[l] = dis_list[r]
			uid_list[l] = uid_list[r]
			while key >= dis_list[l] and l < r: 
				l += 1
			dis_list[r] = dis_list[l]
			uid_list[r] = uid_list[l]
		dis_list[l] = key
		uid_list[l] = tmp
		quickSort(uid_list, dis_list, left, l - 1)
		quickSort(uid_list, dis_list, l + 1, right)

class VectorModel(object):
	def __init__(self):
		self.doc_uid = []				# str
		self.doc_vector = []			# list
		self.content_invert = {}		# pair: {query_content_word : doc_uid set}
		self.global_idf_dict = {}		# pair: {word : idf}
		self.global_vec_index = []

		self.query_qid = []				# str
		self.query_vector = []			# list
		self.query_seg = []				# set

	# path: term_idf.txt
	def getIdf(self, path):
		idf_file = open(path, 'r')
		while 1:
			line = idf_file.readline()
			if not line:
				break
			tmp_list = line.split("\t")
			self.global_idf_dict[tmp_list[0]] = float(tmp_list[1])
		idf_file.close()
		print('getIdf finished')

	# path: ntcir14_test_query.json
	def getQuery(self, path):
		query_doc = open(path, 'r')
		while 1:
			line = query_doc.readline()
			if not line:
				break
			content = json.loads(line)
			keywords = content['query_seg'].split(' ')
			print(content['qid'])
			for word in keywords:
				if word not in self.global_vec_index:
					self.global_vec_index.append(word)
		query_doc.close()
		print('get query keywords and set global index finished')

	# path: ntcir14_test_doc.json
	def getDoc(self, path):
		doc = open(path, 'r')
		ans = 0
		while 1:
			ans += 1
			#print(ans)
			line = doc.readline()
			if not line:
				break
			content = json.loads(line)
			tmp_uid = content['uid']
			tmp_content_list = content['content_seg'].split()
			# judge in query or not
			flag = 0
			tmp_content_dict = {}
			for word in tmp_content_list:
				if word not in self.global_vec_index:
					continue
				else:
					flag = 1
					# set up invert
					if word not in self.content_invert:
						self.content_invert[word] = set()
					self.content_invert[word].add(tmp_uid)
					# set vec
					tmp_content_dict.setdefault(word, 0)
					tmp_content_dict[word] += 1
			if flag == 0:
				continue
			
			tmp_vec = []
			for word in self.global_vec_index:
				if word in tmp_content_dict:
					tmp_vec.append(tmp_content_dict[word])
				else:
					tmp_vec.append(0)
			self.doc_uid.append(tmp_uid)
			self.doc_vector.append(tmp_vec)
			
			#break
		doc.close()
		print('get doc & invert finished')

	def query(self, querypath, outputpath, cutoff):
		query_doc = open(querypath, 'r')
		while 1:
			line = query_doc.readline()
			if not line:
				break
			content = json.loads(line)
			keywords = content['query_seg'].split(' ')
			#print(content['qid'])
			tmp_dict = {}
			for word in keywords:
				tmp_dict.setdefault(word, 0)
				tmp_dict[word] += 1
			tmp_vec = []
			for word in self.global_vec_index:
				if word in tmp_dict:
					tmp_vec.append(tmp_dict[word])
				else:
					tmp_vec.append(0)
			tmp_set = set()
			tmp_set.update(keywords)
			self.query_qid.append(content['qid'])
			self.query_vector.append(tmp_vec)
			self.query_seg.append(tmp_set)
		query_doc.close()
		#print('get query and set vector finished')

		output_file = open(outputpath,'w')
		for i in range(len(self.query_qid)):
		#for i in range(5):
			tar_uid = set()
			uid_list = []
			dis_list = []
			for word in self.query_seg[i]:
				if word in self.content_invert:
					for uid in self.content_invert[word]:
						if uid not in tar_uid:
							tar_uid.add(uid)
							pos = self.doc_uid.index(uid)
							dis = calDistance(self.doc_vector[pos], self.query_vector[i])
							uid_list.append(uid)
							dis_list.append(dis)
			# sort before output
			quickSort(uid_list, dis_list, 0, len(dis_list) - 1)
			length = len(uid_list)
			for j in range(len(uid_list)):
				#print('query', self.query_qid[i], uid_list[length - 1 - j], dis_list[length - 1 - j], file = output_file)
				output_file.write('%s Q0 %s %d %f VectorSpace\n' % (self.query_qid[i], uid_list[length - 1 - j], j + 1, dis_list[length - 1 - j]))
				if j + 1 >= cutoff:
					break
		output_file.close()
	
	def saveModel(self, path):
		with open(path, 'wb') as f:
			pickle.dump(self, f)
			f.close()

	@staticmethod
	def loadModel(modelpath):
		with open(modelpath, 'rb') as f:
			ret = pickle.load(f)
			f.close()
			return ret

if __name__ == "__main__":
	start_time = time.time()
	'''my_model = VectorModel()
	my_model.getIdf('term_idf.txt')
	my_model.getQuery('ntcir14_test_query.json')
	my_model.getDoc('ntcir14_test_doc.json')
	my_model.saveModel('VectorSpaceModel.pkl')'''
	my_model = VectorModel.loadModel('VectorSpaceModel.pkl')
	my_model.query('ntcir14_test_query.json', 'result_10.txt', 10)
	end_time = time.time()
	print('time spend:', end_time - start_time)
