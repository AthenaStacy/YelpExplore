from flask import Flask,render_template,request,redirect,session
app = Flask(__name__)

import pandas as pd
import datetime
import numpy as np
import pickle
from scipy import spatial
import itertools
from scipy import signal
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import urllib2 
import urllib
import operator

pd.options.mode.chained_assignment = None  # default='warn'

@app.route('/overview')
def uploaded_file1():
    filename = 'n_vs_date.png'
    return render_template('overview.html', filename=filename)


@app.route('/hello_page')
def hello_world():
    # this is a comment, just like in Python
    # note that the function name and the route argument
    # do not need to be the same.
    return 'Hello world!'

@app.route('/figure1')
def uploaded_file2():
    filename = 'n_vs_date.png'
    return render_template('graph1.html', filename=filename)

@app.route('/figure2')
def uploaded_file3():
    filename = 'barchart.png'
    return render_template('graph2.html', filename=filename)

nreturn = 20 #number of top words to return

app.question1={}
app.question1['What city or business are you interested in?']=('Business','City')

app.nquestion1=len(app.question1)

app.secret_key = 'DONTCAREWHATTHISIS'

#dir_name = ''
#reviews = pd.read_pickle(dir_name+'reviews_random')

dir_name = 'https://s3-us-west-2.amazonaws.com/yelp-explore/'
reviews = pickle.load(urllib.urlopen(dir_name+'reviews_random'))

transformer = TfidfTransformer() #to do tfidf weighting on all of the bag_of_words vectors


@app.route('/')
@app.route('/index',methods=['GET', 'POST'])
def index():
	session['question1_answered'] = 0
	nquestion1=app.nquestion1
	if request.method == 'GET':
		return render_template('welcome.html',num=nquestion1)
	else:
 		return redirect('/main')

@app.route('/main')
def main():
	if session['question1_answered'] > 0: 
		which_way = business_check()  ##function to check for business in database##
		if(which_way == 1):
			find_top_words()   ##function to find similar songs##
			return render_template('result.html', \
							   input_location=session['input'], \
							   unigram=session['result']['unigram'], \
							   bigram=session['result']['bigram'], \
							   trigram=session['result']['trigram'])
		if(which_way == 2):  #found no MATCHES
			return render_template('tryagain.html')
		if(which_way == 3):  #got an artist, now need to list their songs
			return redirect('/pickaplace')
		if(which_way == 4):  #found multiples songs with that title
			return redirect('/disambiguate')

	else:
		return redirect('/next')


 
@app.route('/next',methods=['GET', 'POST'])
def next(): #remember the function name does not need to match the URL
	if request.method == 'GET':
		session['business_or_city'] = ''
		session['business_name'] = ''
		session['city_name'] = ''
		session['question1_answered'] = 0
		#for clarity (temp variables)
		n2 = app.nquestion1 - len(app.question1) + 1
		q2 = app.question1.keys()[0] #python indexes at 0
		#this will return the answers corresponding to q
		a1, a2= app.question1.values()[0] 
		return render_template('layout2.html',num=n2,question=q2,ans1=a1,ans2=a2)
	else:	#request was a POST
		session['question1_answered'] = 1
		session['business_name'] = request.form['business_name']
		session['city_name'] = request.form['city_name']
		session['business_or_city'] = request.form['answer_from_layout2']
	return redirect('/main')



@app.route('/pickaplace',methods=['GET', 'POST'])
def pickaplace(): #remember the function name does not need to match the URL
	if request.method == 'GET':
		return render_template('pickaplace.html',business_list=session['business_list'], \
		                     address_list=session['address_list'])
	else:	#request was a POST
		session['business_or_city'] = 'Both'
		session['business_name'] = session['business_list'][str(request.form['business_pick'])]
	return redirect('/main')


@app.route('/disambiguate',methods=['GET', 'POST'])
def disambiguate(): #remember the function name does not need to match th eURL
	if request.method == 'GET':
		return render_template('disambiguate.html', match_business=session['match_business'],  \
													match_city=session['match_city'])
	else:	#request was a POST
		session['business_or_city'] = 'Both'
		session['business_name'] = session['match_business'][str(request.form['business_pick'])]
		session['city_name'] = session['match_city'][str(request.form['city_pick'])]
	return redirect('/main')
 
 
def business_check():
	match = {}


	if(session.get('business_or_city') == 'Both'):
		return(1)	


	if(session.get('business_or_city') == 'City'):
		my_city = session['city_name']
		match = reviews[reviews['city']==my_city]
		lmatch = len(match)


	if(session.get('business_or_city') == 'Business'):
		my_business = session['business_name']
		match = reviews[reviews['business_name']==my_business]
		lmatch = len(match)


	if  lmatch>0 and session.get('business_or_city') == 'Business':  #Found one matching business!!
		#session['city_name'] = reviews[reviews['business_name']==my_business].city.values[0]
		return(1)

		
	if lmatch>0 and session.get('business_or_city') == 'City':
		return(1)

	if lmatch==0:  #Found NO matches
		return(2)

#	if(session['business_or_city'] == 'City'): #Got an city, now need to get their songs#
#		print 'Looking for businesses in that city', my_city
#		session['business_list']={}
#		session['address_list']={}
#		business_list = business[business['city']==my_city]
#		business_list = business_list.reset_index()
#		for x in range(len(business_list['name'])):
#			session['business_list'][x] = business_list['name'][x]
#			session['address_list'][x] = business_list['full_address'][x]
#			print session['business_list'][x], session['address_list'][x]
#		return(3)				
	
	if lmatch>0 and session['business_or_city'] == 'Business':  #Found multiple songs
		session['match_city'] = {}
		session['match_business'] = {}
		session['match_address'] = {}
		match = reviews[reviews['business_name']==my_business]
		match = match.reset_index()
		for x in range(len(match['name'])):
			session['match_business'][x] = match['name'][x]
			session['match_city'][x] = match['city'][x]
			session['match_address'][x] = match['full_address'][x]
		return(4)
			



def find_top_words():
	my_business = session['business_name']
	my_city = session['city_name']

	if session['business_or_city'] == 'City':	
		session['input'] = my_city
		reviews_subset = reviews[reviews['city'] == my_city]
	elif session['business_or_city'] == 'Business':
		session['input'] = my_business
		reviews_subset = reviews[reviews['business_name'] == my_business]
	else:
		session['input'] = my_city
		reviews_subset = reviews[1:500]

	reviewsR = reviews[1:10]
	
	reviews_cleanR=[]
	nrev = reviewsR['text_clean'].size
	for i in range( 0, nrev ):
		reviews_cleanR.append(reviewsR['text_clean'].iloc[i])

	
	reviews_clean_subset=[]
	nrev_alt = reviews_subset['text_clean'].size
	nrev = 200
	if nrev_alt < nrev:
		nrev = nrev_alt
	for i in range( 0, nrev ):
		reviews_clean_subset.append(reviews_subset['text_clean'].iloc[i])

	mystringR = ''.join(reviews_cleanR)
	mystring_subset = ''.join(reviews_clean_subset)

	reviews_total = []
	reviews_total.append(mystringR)
	reviews_total.append(mystring_subset)

	reviews_vec = CountVectorizer(analyzer = "word",   \
                             tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = None,   \
                             max_features = 200, ngram_range=(1,1)) 
	reviews_features = reviews_vec.fit_transform(reviews_total)
	reviews_tfidf = transformer.fit_transform(reviews_features)
	reviews_features = reviews_features.toarray()
	reviews_tfidf = reviews_tfidf.toarray()


	reviews2_vec = CountVectorizer(analyzer = "word",   \
                             tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = None,   \
                             max_features = 200, ngram_range=(2,2)) 
	reviews2_features = reviews2_vec.fit_transform(reviews_total)
	reviews2_tfidf = transformer.fit_transform(reviews2_features)
	reviews2_features = reviews2_features.toarray()
	reviews2_tfidf = reviews2_tfidf.toarray()
	

	reviews3_vec = CountVectorizer(analyzer = "word",   \
	                             tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = None,   \
                             max_features = 200, ngram_range=(3,3)) 
	reviews3_features = reviews3_vec.fit_transform(reviews_total)
	reviews3_tfidf = transformer.fit_transform(reviews3_features)
	reviews3_features = reviews3_features.toarray()
	reviews3_tfidf = reviews3_tfidf.toarray()

	vocab = reviews_vec.get_feature_names()
	vocab2 = reviews2_vec.get_feature_names()
	vocab3 = reviews3_vec.get_feature_names()

	i=1
	dist = reviews_tfidf[i] #one row of word features
	count_dict = dict(zip(vocab, dist))
	sorted_count_dict = sorted(count_dict.items(), key=operator.itemgetter(1),reverse=True)
	word, freq = zip(*sorted_count_dict)

	dist2 = reviews2_tfidf[i] #one row of word features
	count_dict2 = dict(zip(vocab2, dist2))
	sorted_count_dict2 = sorted(count_dict2.items(), key=operator.itemgetter(1),reverse=True)
	word2, freq2 = zip(*sorted_count_dict2)

	dist3 = reviews3_tfidf[i] #one row of word features
	count_dict3 = dict(zip(vocab3, dist3))
	sorted_count_dict3 = sorted(count_dict3.items(), key=operator.itemgetter(1),reverse=True)
	word3, freq3 = zip(*sorted_count_dict3)

	x = word[0:nreturn]
	y = word2[0:nreturn]
	z = word3[0:nreturn]

	result = pd.DataFrame({'unigram' : x, 'bigram' : y, 'trigram' : z})

	resulto = {'unigram':{}, 'bigram':{}, 'trigram':{}}
	for x in range(0,nreturn):
		resulto['unigram'][x] = result['unigram'][x]
		resulto['bigram'][x] = result['bigram'][x]
		resulto['trigram'][x] = result['trigram'][x]
	session['result'] = resulto	


if __name__ == "__main__":
	app.run(port=33508)
		
