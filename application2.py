from flask import Flask,render_template,request,redirect,session, send_file
app = Flask(__name__)

import pandas as pd
import datetime
from datetime import datetime
import numpy as np
import pickle
import string
from scipy import spatial
import itertools
from scipy import signal
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import urllib2 
import urllib
import operator
import re
import dill
import requests
from bs4 import BeautifulSoup

import matplotlib
matplotlib.use('Agg') # this allows PNG plotting
import matplotlib.pyplot as plt
from StringIO import StringIO
from wordcloud import WordCloud,STOPWORDS

from bokeh.layouts import gridplot
from bokeh.plotting import figure, show, output_file
from bokeh.embed import components 
from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource, Range1d, LabelSet, Label, PrintfTickFormatter
from bokeh.charts import Bar
from bokeh.charts.attributes import CatAttr

API_KEY="AIzaSyBQe-vggEelYAQ3LzTtBISKmZ_UOF4R7UE"
from bokeh.models import (
  GMapPlot, GMapOptions, ColumnDataSource, Circle, Triangle, DataRange1d, PanTool, WheelZoomTool, BoxSelectTool
)

pd.options.mode.chained_assignment = None  # default='warn'
dir_name = 'https://s3-us-west-2.amazonaws.com/yelp-explore/'

@app.route('/hello_page')
def hello_world():
    return 'Hello world!'

nreturn_default = 13 #number of top words to return

app.question1={}
app.question1['What category and/or city interests you?']=('Category_typed','Category','City')

app.question2={}
app.question2['What is your Yelp page url?']=('business_url')

app.nquestion1=len(app.question1)

app.option1 = "competition"
app.option2 = "myself"
app.option3 = "predict"

app.attributes_combined = ''
app.reviews = ''

app.secret_key = 'DONTCAREWHATTHISIS'

transformer = TfidfTransformer() #to do tfidf weighting on all of the bag_of_words vectors

@app.route('/')
@app.route('/index',methods=['GET', 'POST'])
def index():
	session['question1_answered'] = 0
	session['question2_answered'] = 0
	nquestion1=app.nquestion1
	if request.method == 'GET':
		return render_template('index.html',num=nquestion1)
	else:
 		return redirect('/main')

@app.route('/main/')
def main():
	if session['question1_answered'] > 0 and session['question2_answered'] == 0 : 
		reviews = load_reviews()
		business = load_business()
		
		business_subset, which_way = get_business_subset(business)
		
		reviews_good, reviews_bad = find_good_and_bad(reviews, 5, 2)

		make_wordcloud(reviews_good, reviews_bad, 'wordcloudG.png', 'wordcloudB.png')

		if(which_way == 1):
			script_topwords, div_topwords = find_top_words(reviews_good, reviews_bad)   ##function to find most frequent words used in reviews

			script_topatts, div_topatts = find_top_atts_and_cats(business_subset)

			if session['refine_by_city'] == 'yes':
				script_map, div_map = city_map(business)
				session['input_map'] = session['input_city']
			else:
				script_map, div_map = country_map(business, business_subset)
				session['input_map'] = session['input_category']+' '+session['input_search']
							
			return render_template('result.html', \
							   script1=script_topwords, div1=div_topwords, \
							   script2=script_topatts, div2=div_topatts, \
							   script4=script_map, div4=div_map, \
							   input_location=session['input_city']+' '+session['input_category']+' '+session['input_search'], \
							   input_map = session['input_map'], \
							   uni_good=session['result']['uni_good'], \
							   tri_good=session['result']['tri_good'], \
							   uni_bad=session['result']['uni_bad'], \
							   tri_bad=session['result']['tri_bad'], \
							   question2 = app.question2.keys()[0], option2 = app.option2)		
							   					   
		if(which_way == 2):  #found no MATCHES
			return render_template('tryagain.html')
		
	else:
		return redirect('/next')

@app.route('/myself/', methods=['GET', 'POST'])
def myself():
 	if request.method == 'GET' and session['question2_answered'] > 0: 
 	
 		url_myself = session['business_url']
 		print 'url_myself =', url_myself
 		
 		review_list = ''
 		star_list = ''
 		date_list = ''
 		#if len(app.attributes_combined) < 1:					
		stars_myself, review_count, app.attributes_combined, date_list, review_list, star_list = load_myself(url_myself)
		session['stars_myself'] = stars_myself
		session['review_count'] = review_count
		session['attributes_myself']= app.attributes_combined

		print len(review_list), len(date_list), len(star_list)

		reviews_myself =  pd.DataFrame({'text_clean' :review_list, \
	                                 'date' : date_list, \
	                                 'stars' : star_list })

		#reviews_myself = load_myreviews()
		#reviews = load_reviews()
		#business = load_business()		
		#business_id = u'McikHxxEqZ2X0joaRNKlaw'  #just put in a random business for now									
		#reviews_myself = pd.read_pickle('/Users/minerva/Desktop/DataIncubator/Yelp_examples/reviews_testcase')
		#business_myself = business[business.index == business_id]
		#session['attributes_myself'] = business_myself['attributes_combined'].iloc[0]
		#session['stars_myself'] = str(business_myself['stars'].iloc[0])
		#session['review_count'] = str(business_myself['review_count'].iloc[0])
			
		#reviews_myself = reviews_myself[1:3]
		#print 'bus_id =', business_id, 'len reviews =', len(reviews_myself)
	
		if len(reviews_myself) < 5:
			reviews_good_myself = ['insufficient number of reviews, but I am sure you are great and excellent']*100
			reviews_good_myself = ' '.join(reviews_good_myself)
			reviews_bad_myself = ['seriously though increase your yelp page review count']*100
			reviews_bad_myself = ' '.join(reviews_bad_myself)
		else:		
			reviews_good_myself, reviews_bad_myself = find_good_and_bad(reviews_myself, 4, 3)
			
		try:
			make_wordcloud(reviews_good_myself, reviews_bad_myself, 'wordcloudG_myself.png', 'wordcloudB_myself.png')
		except:
			pass
		script_topwords, div_topwords = find_top_words(reviews_good_myself, reviews_bad_myself)
		script_stars, div_stars = star_ratings(reviews_myself)

		op3 = app.option3		
			
		return render_template('result_myself.html', \
							   script1=script_topwords, div1=div_topwords, \
							   script2=script_stars, div2=div_stars, \
							   option3 = op3, \
							   stars = session['stars_myself'], \
							   nrev = session['review_count'])
	else:
		return redirect('/next')
 
@app.route('/predict', methods=['GET', 'POST'])
def predict():
	
	#direc = '/Users/minerva/Desktop/DataIncubator/Yelp_examples/'
	direc = ''
	model_myself = dill.load(open(direc+"yelp_full_model", "r"))

	url_myself = session['business_url']
	
	if len(app.reviews) < 1: 
		review_list = load_my_reviews(url_myself)
		app.reviews = ' '.join(review_list)
	
	categories_myself = '' 

	business_myself =  pd.DataFrame({'categories' : categories_myself, \
	                                 'attributes_combined' : session['attributes_myself'], \
	                                 'reviews' : app.reviews, \
	                                 'review_count' : [session['review_count']] })

	stars_predict_current = model_myself.predict(business_myself)
	stars_predict_current = np.asscalar(stars_predict_current[0])

	op3 = app.option3

	if len(session['num_new']) < 1:
		session['num_new'] = '10'

	try:
		num_new = int(session['num_new'])
	except:
		pass
			
	att_to_add = "& " + ' & '.join(session['att_to_add'])
	attributes_myself = session['attributes_myself'] + att_to_add

	perception_to_add = session['perception_to_add']*num_new
	food_to_add = session['perception_to_add']*num_new

	rev_to_add = perception_to_add + food_to_add
	rev_to_add = ' '.join(rev_to_add)
	reviews_myself = app.reviews + rev_to_add

	business_myself_new = business_myself
	business_myself_new['attributes_combined'].iloc[0] = attributes_myself
	business_myself_new['reviews'].iloc[0] = reviews_myself
		
	stars_predict = model_myself.predict(business_myself_new)
	stars_predict = np.asscalar(stars_predict[0])
	return render_template('predict.html', \
							option3 = op3, \
							stars = session['stars_myself'],
							foods = session['food_to_add_ascii'], \
							perceptions = session['perception_to_add_ascii'], \
							atts = session['att_to_add_ascii'], \
							nrev = session['review_count'], \
							stars_predict = stars_predict, \
							stars_predict_current = stars_predict_current, \
							nnew = session['num_new'])
	
 
@app.route('/next',methods=['GET', 'POST'])
def next(): #remember the function name does not need to match the URL
	if request.method == 'GET':
		session['category_search'] = ''
		session['category_name'] = ''
		session['city_name'] = ''
		session['question1_answered'] = 0
		session['question2_answered'] = 0
		session['question3_answered'] = 0
		
		
		session['refine_by_city'] = ''
		session['refine_by_category'] = ''
		session['refine_by_search'] = ''
		
		session['input_city'] = ''
		session['input_category'] = ''
		session['input_search'] = ''
		session['business_url'] = ''

		session['food_to_add'] = 'nothing'
		session['att_to_add'] = 'nothing'
		session['perception_to_add'] = 'nothing'
		session['food_to_add_ascii'] = 'nothing'
		session['att_to_add_ascii'] = 'nothing'	
		session['perception_to_add_ascii'] = 'nothing'
		session['num_new'] = '10'	
		
		#for clarity (temp variables)
		n2 = app.nquestion1 - len(app.question1) + 1
		q1 = app.question1.keys()[0] #python indexes at 0		
		q2 = app.question2.keys()[0]
		
		op1 = app.option1
		op2 = app.option2
		
		return render_template('question.html',question1=q1, question2=q2, option1 = op1, option2 = op2)

	else:	#request was a POST
		if request.form['btn'] == app.option1:  #check out the yelp data
			session['question1_answered'] = 1
			try:
				session['category_search'] = request.form['category_search']
			except:
				pass
			try:
				session['category_name'] = request.form['category_name']
			except:
				pass
			try:
				session['city_name'] = request.form['city_name']
			except:
				pass
			try:
				session['refine_by_city'] = request.form['refine_by_city']
			except:
				pass
			try:
				session['refine_by_category'] = request.form['refine_by_category']
			except:
				pass
			try:
				session['refine_by_search'] = request.form['refine_by_search']
			except:
				pass
		if request.form['btn'] == app.option2:  #check out my own business
			session['question2_answered'] = 1
			session['refine_by_city'] = ''
			session['refine_by_category'] = ''
			session['refine_by_search'] = ''
			try:
				session['business_url'] = request.form['business_url']
			except:
				pass		

		if request.form['btn'] == app.option3:  #predict ratings
			session['question3_answered'] = 1
			session['food_to_add'] = 'nothing'
			session['att_to_add'] = 'nothing'
			session['perception_to_add'] = 'nothing'
			session['food_to_add_ascii'] = 'nothing'
			session['att_to_add_ascii'] = 'nothing'
			session['perception_to_add_ascii'] = 'nothing'
			session['num_new']='10'
			if len(request.form.getlist('food_to_add')) > 0:
				session['food_to_add'] = request.form.getlist('food_to_add')
				session['food_to_add_ascii'] = [x.encode('UTF8') for x in session['food_to_add']]
			if len(request.form.getlist('att_to_add')) > 0:
				session['att_to_add'] = request.form.getlist('att_to_add')
				session['att_to_add_ascii'] = [x.encode('UTF8') for x in session['att_to_add']]
			if len(request.form.getlist('perception_to_add')) > 0:
				session['perception_to_add'] = request.form.getlist('perception_to_add')
				session['perception_to_add_ascii'] = [x.encode('UTF8') for x in session['perception_to_add']]
			try:
				session['num_new'] = request.form['num_new']
			except:
				session['num_new']='10'
				
		if session['question2_answered'] == 1 and session['question3_answered'] == 0:
			return redirect('/myself')

		elif session['question3_answered'] == 1:
			return redirect('/predict')

		else:
			return redirect('/main')
 			
			
if __name__ == "__main__":
	app.run(port=33508)


##########################################################################
##SEPARATE FUNCTIONS FOR DATA PROCESSING START HERE
##########################################################################
##########################################################################
def load_reviews():
	#reviews = pd.read_pickle(dir_name+'reviews_random')
	#reviews = pickle.load(urllib.urlopen(dir_name+'reviews_random'))
	#reviews =  pickle.load(urllib.urlopen(dir_name+'reviews_random_new'))
	reviews = pd.read_pickle('reviews_random_new')

	return reviews

def load_business():
	#business = pickle.load(urllib.urlopen(dir_name+'business_and_rates_and_atts'))
	#business = pd.read_pickle('/Users/minerva/Desktop/DataIncubator/Yelp_examples/business_and_rates_and_atts')
	business = pd.read_pickle('business_and_rates_and_atts')
	return business


#############################################################################
#############################################################################

def get_business_subset(business):

	my_category = session['category_name']
	my_search = session['category_search']
	my_city = session['city_name']

	which_way = 1

	if session['refine_by_city'] == 'yes' and len(my_city) > 1:	
		session['input_city'] = my_city
		business = business[business['city'] == my_city]
		
	if session['refine_by_category'] == 'yes' and len(my_category) > 1:
		session['input_category'] = my_category +' Category'
		my_categories = business['categories'].tolist()
		has_category = list([pick_category(x, my_category) for x in my_categories])	
		business['has_category'] = has_category		
		business = business[business['has_category'] == True]
	
	if session['refine_by_search'] == 'yes' and len(my_search) > 1:
		session['input_search'] = my_search +' Category'
		my_categories = business['categories'].tolist()
		has_category = list([pick_category(x, my_category) for x in my_categories])	
		business['has_category'] = has_category		
		business = business[business['has_category'] == True]
			
	if len(business) < 1:
		which_way = 2
	
	return business, which_way
	
#############################################################################
#############################################################################

def find_good_and_bad(reviews, good_cut, bad_cut):

	my_category = session['category_name']
	my_search = session['category_search']
	my_city = session['city_name']
	
	reduced = 0

	if session['refine_by_city'] == 'yes' and len(my_city) > 1:	
		reviews = reviews[reviews['city'] == my_city]
		reduced = 1
		
	if session['refine_by_category'] == 'yes' and len(my_category) > 1:
		my_categories = reviews['categories'].tolist()
		has_category = list([pick_category(x, my_category) for x in my_categories])	
		reviews['has_category'] = has_category		
		reviews = reviews[reviews['has_category'] == True]
		reduced = 1
	
	if session['refine_by_search'] == 'yes' and len(my_search) > 1:
		session['input_search'] = my_search +' Category'
		my_categories = reviews['categories'].tolist()
		has_category = list([pick_category(x, my_category) for x in my_categories])	
		reviews['has_category'] = has_category		
		reviews = reviews[reviews['has_category'] == True]
		reduced = 1
	

	if reduced == 0:
		reviews = reviews[1:500]

	#Row 0 is 5-star reviews.  Row 1 is 1-star reviews
	reviews_good = reviews[reviews['stars'] >= good_cut]
	reviews_good_list = []
	for i in range( 0, len(reviews_good)):
		reviews_good_list.append(reviews_good['text_clean'].iloc[i])
	reviews_good = ' '.join(reviews_good_list)

	reviews_bad = reviews[reviews['stars'] <= bad_cut]
	reviews_bad_list = []
	for i in range( 0, len(reviews_bad)):
		reviews_bad_list.append(reviews_bad['text_clean'].iloc[i])
	reviews_bad = ' '.join(reviews_bad_list)
	
	return (reviews_good,reviews_bad)


#############################################################################
#############################################################################
def find_top_words(reviews_good, reviews_bad):

	reviews_total = []
	
	reviews_total.append(reviews_good)
	reviews_total.append(reviews_bad)
	
	stop_words = ['still', 'really', 'would', 'also', 'go', 'get']
	reviews_vec = CountVectorizer(analyzer = "word", tokenizer = None, preprocessor = None, \
                    stop_words = stop_words, max_features = 200, ngram_range=(1,1)) 
	reviews_features = reviews_vec.fit_transform(reviews_total)
	reviews_tfidf = transformer.fit_transform(reviews_features)
	reviews_features = reviews_features.toarray()
	reviews_tfidf = reviews_tfidf.toarray()

	reviews2_vec = CountVectorizer(analyzer = "word", tokenizer = None, preprocessor = None, \
                    stop_words = None,  max_features = 500, ngram_range=(2,2)) 
	reviews2_features = reviews2_vec.fit_transform(reviews_total)
	reviews2_tfidf = transformer.fit_transform(reviews2_features)
	reviews2_features = reviews2_features.toarray()
	reviews2_tfidf = reviews2_tfidf.toarray()

	reviews3_vec = CountVectorizer(analyzer = "word", tokenizer = None, preprocessor = None, \
                    stop_words = None,  max_features = 500, ngram_range=(3,3)) 
	reviews3_features = reviews3_vec.fit_transform(reviews_total)
	reviews3_tfidf = transformer.fit_transform(reviews3_features)
	reviews3_features = reviews3_features.toarray()
	reviews3_tfidf = reviews3_tfidf.toarray()

	vocab = reviews_vec.get_feature_names()
	vocab2 = reviews2_vec.get_feature_names()
	vocab3 = reviews3_vec.get_feature_names()
	nreturn = min(nreturn_default, len(vocab), len(vocab2), len(vocab3))

	for i in range(2):
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

		if i == 0:
			ugram_good = word[0:nreturn]
			bgram_good = word2[0:nreturn]
			tgram_good = word3[0:nreturn]
			ufreq_good = freq[0:nreturn]
			bfreq_good = freq2[0:nreturn]
			tfreq_good = freq3[0:nreturn]
			

		if i == 1:
			ugram_bad = word[0:nreturn]
			bgram_bad = word2[0:nreturn]
			tgram_bad = word3[0:nreturn]
			ufreq_bad = freq[0:nreturn]
			bfreq_bad = freq2[0:nreturn]
			tfreq_bad = freq3[0:nreturn]

	result = pd.DataFrame({'uni_good' : ugram_good, 'tri_good' : tgram_good, \
	                       'uni_bad' : ugram_bad, 'tri_bad' : tgram_bad})

	resulto = {'uni_good':{}, 'tri_good':{}, 'uni_bad':{}, 'tri_bad':{}}
	for x in range(0,nreturn):
		resulto['uni_good'][x] = result['uni_good'][x]
		resulto['tri_good'][x] = result['tri_good'][x]
		resulto['uni_bad'][x] = result['uni_bad'][x]
		resulto['tri_bad'][x] = result['tri_bad'][x]		
	session['result'] = resulto	
	
	
	###########################################################################
	#Get relative word counts
	###########################################################################
		
	bar_df =  pd.DataFrame({'Word' : bgram_good, 'Rate' : bfreq_good})
	p1 = Bar(bar_df, label=CatAttr(columns=['Word'], sort=False), 
	        values='Rate', color='green', legend=(700,700),
	        title="Most common 2-word phrases unique to POSITIVE reviews",
	        xlabel = '', ylabel='Relative Frequency', width=600)
	p1.xaxis.axis_label_text_font_size = "20pt"
	p1.yaxis.axis_label_text_font_size = "20pt"
	p1.yaxis.major_label_orientation = "horizontal"
	p1.xaxis.axis_label_text_font_style = "normal"
	p1.yaxis.axis_label_text_font_style = "normal"
	p1.yaxis.major_label_text_font_size = "20pt"
	p1.xaxis.major_label_text_font_size = "11pt"
	p1.xaxis.major_label_text_font_style = "bold"
	p1.title.text_font_size = "12pt"
	p1.yaxis[0].formatter = PrintfTickFormatter(format="%5.3f")

	bar_df =  pd.DataFrame({'Phrase' : tgram_good, 'Rate' : tfreq_good})
	p2 = Bar(bar_df, label=CatAttr(columns=['Phrase'], sort=False), 
	         values='Rate', color='blue', legend=(700,700),
	        title="Most common 3-word phrases unique to POSITIVE reviews",
	        xlabel = '', ylabel='Relative Frequency', width=600)
	p2.xaxis.axis_label_text_font_size = "20pt"
	p2.yaxis.axis_label_text_font_size = "20pt"
	p2.yaxis.major_label_orientation = "horizontal"
	p2.xaxis.axis_label_text_font_style = "normal"
	p2.yaxis.axis_label_text_font_style = "normal"
	p2.yaxis.major_label_text_font_size = "20pt"
	p2.xaxis.major_label_text_font_size = "11pt"
	p2.xaxis.major_label_text_font_style = "bold"
	p2.title.text_font_size = "12pt"
	p2.yaxis[0].formatter = PrintfTickFormatter(format="%5.3f")
	
	bar_df =  pd.DataFrame({'Word' : bgram_bad, 'Rate' : bfreq_bad})
	p3 = Bar(bar_df, label=CatAttr(columns=['Word'], sort=False), 
	        values='Rate', color='magenta', legend=(700,700) ,
	        title="Most common 2-word phrases unique to NEGATIVE reviews",
	        xlabel = '', ylabel = 'Relative Frequency', width=600)
	p3.xaxis.axis_label_text_font_size = "20pt"
	p3.yaxis.axis_label_text_font_size = "20pt"
	p3.yaxis.major_label_orientation = "horizontal"
	p3.xaxis.axis_label_text_font_style = "normal"
	p3.yaxis.axis_label_text_font_style = "normal"
	p3.yaxis.major_label_text_font_size = "15pt"
	p3.xaxis.major_label_text_font_size = "11pt"
	p3.xaxis.major_label_text_font_style = "bold"
	p3.title.text_font_size = "12pt"
	p3.yaxis[0].formatter = PrintfTickFormatter(format="%5.3f")

	bar_df =  pd.DataFrame({'Phrase' : tgram_bad, 'Rate' : tfreq_bad})
	p4 = Bar(bar_df, label=CatAttr(columns=['Phrase'], sort=False), 
	        values='Rate',  legend=(700,700),
	        title="Most common 3-word phrases unique to NEGATIVE reviews",
	        xlabel = '', ylabel = 'Relative Frequency', width=600)
	p4.xaxis.axis_label_text_font_size = "20pt"
	p4.yaxis.axis_label_text_font_size = "20pt"
	p4.yaxis.major_label_orientation = "horizontal"
	p4.xaxis.axis_label_text_font_style = "normal"
	p4.yaxis.axis_label_text_font_style = "normal"
	p4.yaxis.major_label_text_font_size = "20pt"
	p4.xaxis.major_label_text_font_size = "11pt"
	p4.xaxis.major_label_text_font_style = "bold"
	p4.title.text_font_size = "12pt"
	p4.yaxis[0].formatter = PrintfTickFormatter(format="%5.3f")
		
	plots1 = row(p1, p2)
	plots2 = row(p3, p4)
	#plots = column(plots1, plots2)
	plots = row(p2, p4)
	script, div = components(plots)
	return(script, div)
	
	
#############################################################################
#############################################################################

def make_wordcloud(reviews_good, reviews_bad, fname_good, fname_bad):

	wordcloud_good = WordCloud(max_font_size=40, relative_scaling=.5 , 
		                 stopwords=STOPWORDS).generate(reviews_good)
	wordcloud_bad = WordCloud(max_font_size=40, relative_scaling=.5 , 
		                 stopwords=STOPWORDS).generate(reviews_bad)	
                
	plt.imshow(wordcloud_good)
	plt.title("Positive reviews")
	plt.axis('off')
	plt.savefig('static/'+fname_good, bbox_inches='tight', transparent = True)

	plt.imshow(wordcloud_bad)
	plt.title("Negative reviews")
	plt.axis('off')
	plt.savefig('static/'+fname_bad, bbox_inches='tight', transparent = True)

#############################################################################
#############################################################################

#############################################################################
#############################################################################
def pick_category(x, cat_name): 
	if cat_name in x: 
		return True
	else:
		return False

def map_cuisine(x): 
    found = 0
    if 'American' in x or "Steakhouse" in x or 'Burger' in x or 'Sandwich' in x: 
        found = 1
        return 'DarkGreen'
    if 'Mexican' in x:
        found = 1
        return 'RoyalBlue'
    if 'Italian' in x:
        found = 1
        return 'Gold'
    if 'Chinese' in x:
        found =1
        return 'FireBrick'
    if 'Japanese' in x or 'Sushi' in x: 
        found = 1
        return 'Orange'
    if found == 0:
        return 'clear'
  
def map_attribute(x): 
    found = 0
    if "casual" in x: 
        found = 1
        return 'DarkGreen'
    if 'trendy' in x or 'upscale' in x or 'classy' in x:
        found = 1
        return 'RoyalBlue'
    if 'hipster' in x or 'divey' in x:
        found = 1
        return 'Gold'
    if 'touristy' in x:
        found =1
        return 'FireBrick'
    if found == 0:
        return 'clear'
        
def map_star(x): 
    if x >= 4.5: 
        return 'green'
    if x >= 3.5 and x < 4.5:
        return 'blue'
    if x >= 2.5 and x < 3.5:
        return 'yellow'
    if x < 2.5:
        return 'red'

#############################################################################
#############################################################################

def att_to_dict(x):        
	att_list = x
	att_list = att_list.lstrip("&\s+")
	att_list = att_list.rstrip()
	att_list = re.split('\s*&\s*', att_list)
	one_array = np.ones(len(att_list))
	x_out = {cat_list:one_array for cat_list, one_array in zip(att_list, one_array)}
	return x_out

def cat_to_dict(x):
	cat_list = x
	cat_list = re.sub("'", "", cat_list)
	cat_list = re.sub("\&", ",", cat_list)
	cat_list = cat_list.strip('\'')
	cat_list = cat_list.strip('\"')
	cat_list = cat_list.strip('[')
	cat_list = cat_list.strip(']')
	cat_list = re.split(',\s+', cat_list)
	one_array = np.ones(len(cat_list))
	x_out = {cat_list:one_array for cat_list, one_array in zip(cat_list, one_array)}
	return x_out
#############################################################################
#############################################################################
def find_top_atts_and_cats(business_subset):
	my_attributes = business_subset['attributes_combined'].tolist()
	att_dict = list(map(att_to_dict, my_attributes))
	v = DictVectorizer(sparse=False)
	X = v.fit_transform(att_dict)
	vocab = v.get_feature_names()

	att_count = np.zeros(X.shape[1])

	for i in range(len(vocab)):
		att_count[i] = X[:,i].sum()  

	count_dict = dict(zip(vocab, att_count))   
	sorted_count_dict = sorted(count_dict.items(), key=operator.itemgetter(1),reverse=True)
	att, freq = zip(*sorted_count_dict)
	att_list = att[1:21]

	star_list = np.zeros(len(att_list))

	i=0
	for my_attribute in att_list:
		has_attribute = []
		has_attribute = list([pick_category(x, my_attribute) for x in my_attributes])	
		business_subset['has_attribute'] = has_attribute		
		business_subset_new = business_subset[business_subset['has_attribute'] == True]
		star_list[i] = business_subset_new['stars'].mean()
		i = i+1

	count_dict = dict(zip(att_list, star_list)) 
	sorted_count_dict = sorted(count_dict.items(), key=operator.itemgetter(1),reverse=True)
	att_list, star_list = zip(*sorted_count_dict)
		
	bar_df =  pd.DataFrame({'Attribute' : att_list, 'Avg. Star Rating' : star_list})
	
	p1 = Bar(bar_df, label=CatAttr(columns=['Attribute'], sort=False), values='Avg. Star Rating', 
	        color='green', legend=(700,700),
	        title="Most frequent attributes & avg. rating of stores with these attributes",
	        xlabel = '', ylabel='Average Star Rating ', width=600)
	p1.xaxis.axis_label_text_font_size = "20pt"
	p1.yaxis.axis_label_text_font_size = "20pt"
	p1.xaxis.axis_label_text_font_style = "normal"
	p1.yaxis.axis_label_text_font_style = "normal"
	p1.yaxis.major_label_text_font_size = "20pt"
	p1.xaxis.major_label_text_font_size = "11pt"
	p1.xaxis.major_label_text_font_style = "bold"
	p1.title.text_font_size = "11pt"	
	p1.yaxis[0].formatter = PrintfTickFormatter(format="%5.2f")

    ##############################################################################
    ##############################################################################
	my_categories = business_subset['categories'].tolist()
	cat_dict = list(map(cat_to_dict, my_categories))
	v = DictVectorizer(sparse=False)
	X = v.fit_transform(cat_dict)
	vocab = v.get_feature_names()

	cat_count = np.zeros(X.shape[1])

	for i in range(len(vocab)):
		cat_count[i] = X[:,i].sum()  

	count_dict = dict(zip(vocab, cat_count))   
	sorted_count_dict = sorted(count_dict.items(), key=operator.itemgetter(1),reverse=True)
	categ, freq = zip(*sorted_count_dict)

	cat_list = categ[0:20]

	star_list = np.zeros(len(cat_list))

	i=0
	for my_category in cat_list:
		has_category = []
		has_category = list([pick_category(x, my_category) for x in my_categories])	
		business_subset['has_category'] = has_category	
		business_subset_new = business_subset[business_subset['has_category'] == True]
		star_list[i] = business_subset_new['stars'].mean()
		i = i+1

	count_dict = dict(zip(cat_list, star_list)) 
	sorted_count_dict = sorted(count_dict.items(), key=operator.itemgetter(1),reverse=True)
	cat_list, star_list = zip(*sorted_count_dict)
		
	bar_df =  pd.DataFrame({'Category' : cat_list, 'Avg. Star Rating' : star_list})
	
	p2 = Bar(bar_df, label=CatAttr(columns=['Category'], sort=False), values='Avg. Star Rating',  
	        color='blue', legend=(700,700),
	        title="Frequent categories & avg. rating of stores under your search",
	        xlabel = '', ylabel='Average Star Rating ', width=600)
	p2.xaxis.axis_label_text_font_size = "20pt"
	p2.yaxis.axis_label_text_font_size = "20pt"
	p2.xaxis.axis_label_text_font_style = "normal"
	p2.yaxis.axis_label_text_font_style = "normal"
	p2.yaxis.major_label_text_font_size = "20pt"
	p2.xaxis.major_label_text_font_size = "11pt"
	p2.xaxis.major_label_text_font_style = "bold"
	p2.title.text_font_size = "11pt"	
	p2.yaxis[0].formatter = PrintfTickFormatter(format="%5.2f")
	
	plots = row(p1,p2)
	script, div = components(plots)
	return(script, div)
	
#############################################################################
#############################################################################

def city_map(business):
	my_city = session['city_name']
	
	business_subset = business[business['city'] == my_city]
	
	lat_avg = business_subset['latitude'].mean()
	lon_avg = business_subset['longitude'].mean()
	my_lat = business_subset['latitude']
	my_lon = business_subset['longitude']
	my_stars =  business_subset['stars']
	star_avg = business_subset['stars'].mean()

	my_colors_stars = list(map(map_star, my_stars))
	my_lat = np.array(my_lat)
	my_lon = np.array(my_lon)
	my_stars = np.array(my_stars)

	
	my_categories = business_subset['categories'].tolist()
	my_colors_cuisine = list(map(map_cuisine, my_categories))

	map_options = GMapOptions(lat=lat_avg, lng=lon_avg, map_type="roadmap", zoom=11)

	plot = GMapPlot(
   	 x_range=DataRange1d(), y_range=DataRange1d(), map_options=map_options, api_key=API_KEY
	)
	plot.title.text = my_city + ' business ratings. Average rating out of five stars = ' + str(star_avg)
	my_colors = my_colors_stars
	source = ColumnDataSource( data=dict(lat = my_lat, lon = my_lon, stars = my_stars, the_colors=my_colors))
	circle = Circle(x="lon", y="lat", size=5, fill_color="the_colors", fill_alpha=0.2, line_color=None)
	plot.add_glyph(source, circle)
	label1 = Label(x=410, y=530, x_units='screen', y_units='screen', text='green:  5 stars', render_mode='css', border_line_color='white', background_fill_color='white')
	label2 = Label(x=410, y=510, x_units='screen', y_units='screen', text='blue:   4 stars', render_mode='css', border_line_color='white', background_fill_color='white')
	label3 = Label(x=410, y=490, x_units='screen', y_units='screen', text='yellow: 3 stars', render_mode='css', border_line_color='white', background_fill_color='white')
	label4 = Label(x=410, y=470, x_units='screen', y_units='screen', text='red:  < 2 stars', render_mode='css', border_line_color='white', background_fill_color='white')
	plot.add_layout(label1)
	plot.add_layout(label2)
	plot.add_layout(label3)
	plot.add_layout(label4)
	plot.add_tools(PanTool(), WheelZoomTool(), BoxSelectTool())

	plot2 = GMapPlot(
   	 x_range=DataRange1d(), y_range=DataRange1d(), map_options=map_options, api_key=API_KEY
	)
	plot2.title.text = my_city + ' locations of various cuisines'
	my_colors = my_colors_cuisine
	source = ColumnDataSource( data=dict(lat = my_lat, lon = my_lon, stars = my_stars, the_colors=my_colors))
	circle = Triangle(x="lon", y="lat", size=5, fill_color="the_colors", fill_alpha=0.2, line_color=None)
	plot2.add_glyph(source, circle)
	label1 = Label(x=410, y=530, x_units='screen', y_units='screen', text='green: American', render_mode='css', border_line_color='white', background_fill_color='white')
	label2 = Label(x=410, y=510, x_units='screen', y_units='screen', text='blue:  Mexican', render_mode='css', border_line_color='white', background_fill_color='white')
	label3 = Label(x=410, y=490, x_units='screen', y_units='screen', text='red: Italian', render_mode='css', border_line_color='white', background_fill_color='white')
	label4 = Label(x=410, y=470, x_units='screen', y_units='screen', text='yellow: Chinese', render_mode='css', border_line_color='white', background_fill_color='white')
	label5 = Label(x=410, y=450, x_units='screen', y_units='screen', text='brown: Japanese', render_mode='css', border_line_color='white', background_fill_color='white')
	plot2.add_layout(label1)
	plot2.add_layout(label2)
	plot2.add_layout(label3)
	plot2.add_layout(label4)
	plot2.add_layout(label5)
	plot2.add_tools(PanTool(), WheelZoomTool(), BoxSelectTool())

	plots = row(plot, plot2)
	script, div = components(plots)
	return script, div

#############################################################################
#############################################################################

def country_map(business, business_subset):
	my_category = session['category_name']
	if session['category_or_city'] == 'Category_typed':
		my_category = session['category_typed']
	lat_avg = business_subset['latitude'].mean()
	lon_avg = business_subset['longitude'].mean()
	my_lat = business_subset['latitude']
	my_lon = business_subset['longitude']
	my_stars =  business_subset['stars']
	star_avg = business_subset['stars'].mean()

	my_colors_stars = list(map(map_star, my_stars))
	my_lat = np.array(my_lat)
	my_lon = np.array(my_lon)
	my_stars = np.array(my_stars)

	map_options = GMapOptions(lat=lat_avg, lng=lon_avg, map_type="roadmap", zoom=4)

	plot = GMapPlot(
   	 x_range=DataRange1d(), y_range=DataRange1d(), map_options=map_options, api_key=API_KEY
	)
	plot.title.text = 'Locations and Star ratings of ' + my_category + ' category. Average rating = ' + str(star_avg)
	my_colors = my_colors_stars
	source = ColumnDataSource( data=dict(lat = my_lat, lon = my_lon, stars = my_stars, the_colors=my_colors))
	circle = Circle(x="lon", y="lat", size=5, fill_color="the_colors", fill_alpha=0.2, line_color=None)
	plot.add_glyph(source, circle)
	label1 = Label(x=410, y=530, x_units='screen', y_units='screen', text='green: 5 stars', render_mode='css', border_line_color='white', background_fill_color='white')
	label2 = Label(x=410, y=510, x_units='screen', y_units='screen', text='blue:  4 stars', render_mode='css', border_line_color='white', background_fill_color='white')
	label3 = Label(x=410, y=490, x_units='screen', y_units='screen', text='yellow: 3 stars', render_mode='css', border_line_color='white', background_fill_color='white')
	label4 = Label(x=410, y=470, x_units='screen', y_units='screen', text='red:  < 2 stars', render_mode='css', border_line_color='white', background_fill_color='white')
	plot.add_layout(label1)
	plot.add_layout(label2)
	plot.add_layout(label3)
	plot.add_layout(label4)
	plot.add_tools(PanTool(), WheelZoomTool(), BoxSelectTool())

	#################################################
	#################################################
	
	lat_avg = business['latitude'].mean()
	lon_avg = business['longitude'].mean()
	my_lat = business['latitude']
	my_lon = business['longitude']
	my_lat = np.array(my_lat)
	my_lon = np.array(my_lon)
	
	my_attributes = business['attributes_combined'].tolist()
	my_colors_att = list(map(map_attribute, my_attributes))
	my_colors = my_colors_att
		
	plot2 = GMapPlot(
   	 x_range=DataRange1d(), y_range=DataRange1d(), map_options=map_options, api_key=API_KEY
	)
	plot2.title.text = 'Locations of Businesses with other Various Attributes'
	source = ColumnDataSource( data=dict(lat = my_lat, lon = my_lon, stars = my_stars, the_colors=my_colors))
	circle = Triangle(x="lon", y="lat", size=5, fill_color="the_colors", fill_alpha=0.2, line_color=None)
	plot2.add_glyph(source, circle)
	label1 = Label(x=410, y=530, x_units='screen', y_units='screen', text='green: casual', render_mode='css', border_line_color='white', background_fill_color='white')
	label2 = Label(x=410, y=510, x_units='screen', y_units='screen', text='blue:  upscale', render_mode='css', border_line_color='white', background_fill_color='white')
	label3 = Label(x=410, y=490, x_units='screen', y_units='screen', text='yellow: touristy', render_mode='css', border_line_color='white', background_fill_color='white')
	label4 = Label(x=410, y=470, x_units='screen', y_units='screen', text='red:  hipster', render_mode='css', border_line_color='white', background_fill_color='white')
	plot2.add_layout(label1)
	plot2.add_layout(label2)
	plot2.add_layout(label3)
	plot2.add_layout(label4)
	plot2.add_tools(PanTool(), WheelZoomTool(), BoxSelectTool())

	
	plots = row(plot, plot2)
	script, div = components(plots)
	return script, div

#############################################################################
#Now to plot stuff for individual businesses!
#############################################################################
	
def star_ratings(reviews):

	stars_myself = reviews['stars']
	stars_myself = np.array(stars_myself)
	star_avg = stars_myself.mean()

	Date = reviews['date']

	bar_df =  pd.DataFrame({'Date' : Date, 'Stars' : stars_myself})

	p = Bar(bar_df, 'Stars', values = 'Stars', xlabel = 'Star Rating out of 5', legend = (700,700),\
        ylabel = 'Number of Reviews', agg='count', title="Star Rating Distribution, Average = "+str(star_avg))
	p.xaxis.axis_label_text_font_size = "20pt"
	p.yaxis.axis_label_text_font_size = "20pt"
	p.xaxis.axis_label_text_font_style = "normal"
	p.yaxis.axis_label_text_font_style = "normal"
	p.yaxis.major_label_text_font_size = "20pt"
	p.xaxis.major_label_text_font_size = "20pt"
	p.xaxis.major_label_text_font_style = "bold"
	p.title.text_font_size = "15pt"

	begin_date = min(reviews['date'])
	end_date = max(reviews['date'])
	datelist = np.arange(begin_date, '2020-01-01', dtype='datetime64[D]')	
	#datelist = np.arange(begin_date, '2020-01-01', dtype='datetime64[D]')
	next_date = begin_date

	#print "1", datelist

	i=1
	while next_date < end_date:
		datelist[i] = next_date
		next_date = next_date + pd.DateOffset(months=+3, days=0)
		i = i+1

	#print "2", datelist

	datelist = datelist[0:(i-1)]
    
	dlen = len(datelist)

	stars_over_time = np.zeros(dlen-1)
	reviews_over_time = np.zeros(dlen-1)

	for i in range(0,dlen-1):
		this_month = reviews[(reviews['date'] > datelist[i]) & (reviews['date'] <= datelist[i+1])]
		running_total = reviews[(reviews['date'] <= datelist[i+1])]
		stars_over_time[i] = running_total['stars'].mean()
		reviews_over_time[i] = len(running_total)

	#print "3", datelist

	datelist = datelist[0:dlen-1]

	p2 = figure(title='Star rating over time', \
              x_axis_label='date', x_axis_type='datetime', y_axis_label = 'Stars')

	p2.line(datelist, stars_over_time, line_width=4)
	p2.xaxis.axis_label_text_font_size = "20pt"
	p2.yaxis.axis_label_text_font_size = "20pt"
	p2.xaxis.axis_label_text_font_style = "normal"
	p2.yaxis.axis_label_text_font_style = "normal"
	p2.yaxis.major_label_text_font_size = "20pt"
	p2.xaxis.major_label_text_font_size = "15pt"
	p2.xaxis.major_label_text_font_style = "bold"
	p2.title.text_font_size = "15pt"

	p3 = figure(title='Total review number over time', \
              x_axis_label='date', x_axis_type='datetime', y_axis_label = 'Review number')

	p3.line(datelist, reviews_over_time, line_width=4)
	p3.xaxis.axis_label_text_font_size = "20pt"
	p3.yaxis.axis_label_text_font_size = "20pt"
	p3.xaxis.axis_label_text_font_style = "normal"
	p3.yaxis.axis_label_text_font_style = "normal"
	p3.yaxis.major_label_text_font_size = "20pt"
	p3.xaxis.major_label_text_font_size = "15pt"
	p3.xaxis.major_label_text_font_style = "bold"
	p3.title.text_font_size = "15pt"


	#plots = p
	plots_row = row(p,p2)
	plots = column(plots_row,p3)
	script, div = components(plots)
	return script, div

#############################################################################
#REVIEW CLEANING
##############################################################################
def review_to_words( raw_review ):
	# Cleaning technique learned from Kaggle Bag of Words competition:
    # Function to convert a raw review to a string of words
    #
    #Remove HTML
	review_text = BeautifulSoup(raw_review).get_text() 
    
    #Remove non-letters        
	letters_only = re.sub("[^a-zA-Z]", " ", review_text)
    
    #Convert to lower case, split into individual words
	words = letters_only.lower().split()                             
 
    #stops = set(stopwords.words("english"))                  
	stops = ['a', 'an', 'the', 'and', 'it', 'but', 'i', 'to', 'me', 'you', 'he', 'she', 
	         'they', 'we', 'was', 'is', 'are', 'of', 'or', 'in', 'has', 'for', 'his', 'her',
	         'my', 'with']

    #Remove stop words
	meaningful_words = [w for w in words if not w in stops]   

	return( " ".join( meaningful_words )) 

#############################################################################
#WEB SCRAPING
#############################################################################
def load_myself(url_myself):

	response = requests.get(url_myself)
	soup = BeautifulSoup(response.text)
	page_block = soup.find(["div"], attrs={"class": 'page-of-pages arrange_unit arrange_unit--fill'})
	npage = re.sub('Page 1 of ', '', page_block.text)
	npage = int(npage)
	
	stars_myself = soup.find(itemprop = 'ratingValue')['content']
	stars_myself = float(stars_myself)

	review_count = soup.find(itemprop = 'reviewCount').text
	review_count = int(review_count)

	attributes_here = soup.find_all(['d1','dt'])
	attributes_def = soup.find_all(['d1','dd'])

	att_here = ''
	for i in range(len(attributes_here)):
		ahere = attributes_here[i].text.strip()
		adef = attributes_def[i].text.strip()
		if adef != 'Yes' and adef != 'No' and adef != 'yes' and adef != 'no':
			x = ahere + '.' + adef.lower()
		else:
			x = ahere
		att_here = att_here + " & " + x

	url_list = []
	npage_here = min(npage-1, 5)
	startnum = 20
	url_list.append(url_myself)

	for i in range(npage_here):
	    next_url = url_myself + '?start=' + str(startnum)
    	url_list.append(next_url)
    	startnum = startnum + 20
	
	review_list = []
	date_list = []
	star_list = []

	for u in url_list:
		print u
		response = requests.get(u)
		soup = BeautifulSoup(response.text)
		reviews = soup.find_all(["p"], attrs={"itemprop": 'description'})
		for r in reviews:
			rtext = r.text
			review_list.append(rtext)
       	 
		dates_all = soup.find_all(itemprop = 'datePublished')
		for d in dates_all:
			date_here = d['content']
			date = datetime.strptime(date_here , '%Y-%m-%d')#.date()
			date_list.append(date)
                        
		stars_all = soup.find_all(itemprop = 'ratingValue')
		for s in stars_all[1:]:
			stars = s['content']
			star_list.append(float(stars))

	reviews_clean = list(map(review_to_words, review_list))
       	
	return stars_myself, review_count, att_here, date_list, reviews_clean, star_list
	
###################################################################################
#more webscraping!
##################################################################################
def load_my_reviews(url_myself):

	response = requests.get(url_myself)
	soup = BeautifulSoup(response.text)
	page_block = soup.find(["div"], attrs={"class": 'page-of-pages arrange_unit arrange_unit--fill'})
	npage = re.sub('Page 1 of ', '', page_block.text)
	npage = int(npage)
	
	url_list = []
	npage_here = min(npage-1, 5)
	startnum = 20
	url_list.append(url_myself)

	for i in range(npage_here):
	    next_url = url_myself + '?start=' + str(startnum)
    	url_list.append(next_url)
    	startnum = startnum + 20
	
	review_list = []

	for u in url_list:
		print u
		response = requests.get(u)
		soup = BeautifulSoup(response.text)
		reviews = soup.find_all(["p"], attrs={"itemprop": 'description'})
		for r in reviews:
			rtext = r.text
			review_list.append(rtext)
       	 
	reviews_clean = list(map(review_to_words, review_list))
      	
	return reviews_clean
	
	
	
	
	
	
