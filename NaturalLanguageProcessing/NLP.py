	



#---Building a spam detection filter---#
def NLP():
	#nltk.download_shell() #dowloading stopwords package from shell
	#www.archive.ics.uci.edu/ml/datasets/

	messages = [line.rstrip() for line in open('SMSSpamCollection')]
	#print(len(messages))
	#print(messages[20])


	messages_df = pd.read_csv('SMSSpamCollection', sep='\t', names=['label', 'message'])
	#print(messages_df.head())
	#print(messages_df.describe())
	#print(messages_df.groupby('label').describe())

	messages_df['length'] = messages_df['message'].apply(len)
	#print(messages_df.head)

	messages_df['length'].plot.hist(bins=100)


	samp_mess = 'Sample message! Notice: there is punctuation.'
	rm_stpwrds = process_text(samp_mess)
	
	#print(messages_df['message'].head(3).apply(process_text))


	#Bag of Words transformer. Create a matrix with all the words/frequencies
	bow_transformer = CountVectorizer(analyzer=process_text).fit(messages_df['message'])
	#print(len(bow_transformer.vocabulary_))

	messages_bow = bow_transformer.transform(messages_df['message'])
	#print('Shape of Sparse Matrix: ', messages_bow.shape)
	#print messages_bow.nnz

	sparsity = (100.0 * messages_bow.nnz / (messages_bow.shape[0] * 
		messages_bow.shape[1]))
	#print(sparsity)

	tf_transformer = TfidfTransformer().fit(messages_bow)

	print(tf_transformer.idf_[bow_transformer.vocabulary_['university']])
	tf_messages = tf_transformer.transform(messages_bow)
	#print(tf_messages)
	spam_detect_model = MultinomialNB().fit(tf_messages, messages_df['label'])
	all_predict = spam_detect_model.predict(tf_messages)
	#print(all_predict)


	msg_train, msg_test, label_train, label_test = train_test_split(messages_df['message'],
		messages_df['label'], test_size=0.3)

	pipeline = Pipeline([
			('bow', CountVectorizer(analyzer=process_text)),
			('tfidf', TfidfTransformer()),
			('classifier', MultinomialNB())
		])
	pipeline.fit(msg_train, label_train)
	predictions = pipeline.predict(msg_test)
	print(classification_report(label_test, predictions))



	#plt.show()



#---Text Preprocessing---#
def process_text(mess):
	nopunc = [m for m in mess if m not in string.punctuation]
	nopunc = ''.join(nopunc)
	nopunc = nopunc.split()

	rm_stpwrds = [word for word in nopunc if word.lower() not in stopwords.words('english')]
	return rm_stpwrds






NLP()