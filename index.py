from flask import *
import os



from python_speech_features import mfcc
import scipy.io.wavfile as wav

import pickle
import operator

import numpy as np

from collections import defaultdict



up_fl='C:\\Users\\SHWETA\\Desktop\\Music_Genre_Code\\mg_dply\\uploaded_files'

allowed_extension = {'wav'}

app=Flask(__name__)

app.secret_key='scrt'

app.config['up_fl']=up_fl

app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024

output="None"



dataset = []

@app.route('/index')
def index():
  return render_template('index.html')

@app.route('/index2')
def index2():
  return render_template('index2.html')

@app.errorhandler(413)
def page_not_found(e):
   flash("File should be less than 2 MB")
   return render_template('index.html', output=output)

@app.route('/', methods=['GET', 'POST'])
def genre():
	global output

	if request.method=='POST':
		f=request.files['file']
		if allowed_file(f.filename):
			f.save(os.path.join(app.config['up_fl'], f.filename))



			loadDataset("my_3sec.dat")

			results=defaultdict(int)
			gnr=['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

			i=1
			for g in gnr:
				results[i]=g
				i+=1
			
			(rate,sig)=wav.read('./uploaded_files/'+f.filename)
			mfcc_feat=mfcc(sig,rate,winlen=0.020,appendEnergy=False)
			covariance = np.cov(np.matrix.transpose(mfcc_feat))
			mean_matrix = mfcc_feat.mean(0)
			feature=(mean_matrix,covariance,0)

			pred=nearestClass(getNeighbors(dataset ,feature , 5))

			ans=results[pred]
			output=ans.capitalize()

			os.remove(os.path.join(app.config['up_fl'], f.filename))			
		else:
			if f.filename == '':
				flash('No file selected')
			else:
				flash('File should be in .wav format')
		
	return render_template('index.html', output=output)



def loadDataset(filename):
	with open("my_3sec.dat" , 'rb') as f:
		while True:
			try:
				dataset.append(pickle.load(f))
			except EOFError:
				f.close()
				break

def distance(instance1 , instance2 , k ):
	distance =0 
	mm1 = instance1[0] 
	cm1 = instance1[1]
	mm2 = instance2[0]
	cm2 = instance2[1]
	distance = np.trace(np.dot(np.linalg.inv(cm2), cm1)) 
	distance+=(np.dot(np.dot((mm2-mm1).transpose() , np.linalg.inv(cm2)) , mm2-mm1 )) 
	distance+= np.log(np.linalg.det(cm2)) - np.log(np.linalg.det(cm1))
	distance-= k
	return distance

def getNeighbors(trainingSet , instance , k):
	distances =[]
	for x in range (len(trainingSet)):
		dist = distance(trainingSet[x], instance, k )+ distance(instance, trainingSet[x], k)
		distances.append((trainingSet[x][2], dist))
	distances.sort(key=operator.itemgetter(1))
	neighbors = []
	for x in range(k):
		neighbors.append(distances[x][0])
	return neighbors  

def nearestClass(neighbors):
	classVote ={}
	for x in range(len(neighbors)):
		response = neighbors[x]
		if response in classVote:
			classVote[response]+=1 
		else:
			classVote[response]=1 
	sorter = sorted(classVote.items(), key = operator.itemgetter(1), reverse=True)
	return sorter[0][0]

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extension



if __name__=='__main__':
	app.run(debug=True)