
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

		
