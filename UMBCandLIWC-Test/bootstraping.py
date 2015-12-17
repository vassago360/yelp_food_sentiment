



posSeeds = [[foodItemID, restaurantID]...]
negSeeds = [[foodItemID, restaurantID]...]



#Read data to get unigrams...
tfidf_transformer = TfidfTransformer()
count_vect = CountVectorizer()
X_traindataResponse_counts = count_vect.fit_transform(traindataResponse)
unigrams = X_traindataResponse_counts.toarray()


kVal = 10
kBest = SelectKBest(f_regression, k=kVal).fit(trainData, trainTarget)
bestFeaturesIndex = kBest.get_support(indices=True)
trainDataKBestFeatures = kBest.transform(trainData)


if __name__ == '__main__':
    pass
    
