import time

import nltk
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords
from sklearn.decomposition import LatentDirichletAllocation,PCA
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, pairwise_distances, calinski_harabaz_score, davies_bouldin_score
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
import matplotlib.pyplot as plt
import porter
import string
from nltk.stem.porter import *
import pandas as pd

eng_stopwords = stopwords.words('english')

dict = []
def prePro():
    with open("cnnhealth.txt", "rb") as filestream:
        terms = 0
        p = PorterStemmer()
        fileContent = filestream.readlines()
        for Documents, line in enumerate(fileContent):
            st = line.decode("utf-8").strip("\n").strip("\r").lower()
            st = re.sub(r'http\S+', '', st)
            mylist = st.split('|')[2]
            mylist = mylist.translate(str.maketrans('','', string.punctuation))
            mylist = p.stem(mylist)
            #print(mylist)

            words = mylist.split()
            words = [w for w in words if w not in eng_stopwords]

            mylist = ' '.join(words)
            #print(mylist)
            terms =terms + len(words)
            dict.append(mylist)
        return dict

        #print("Documents: " + str(Documents) + ", Term Tokens: " + str(terms) + ", Unique Term: "+  str(0) +", Avg Terms Per Document: " + str( round((terms/Documents), 2)) )


def toVector(content,n_components):
    vectorizer = CountVectorizer()
    vector = vectorizer.fit_transform(content)
    vector = vector.toarray()
    import numpy
    a = numpy.asarray(vector)
    numpy.savetxt("bow.csv", a, delimiter="\n")
    print(vector)
    lda = LatentDirichletAllocation(n_components=n_components)
    result = lda.fit_transform(vector)
    return result

def clusterNum(result, df, n_clusters_):

    cluster = []
    tweets = []
    clusterTitles = result
    dbscandf = df
    dbscandf['cluster'] = clusterTitles

    #print(dict)
    #print(dbscandf[dbscandf['cluster'] == 3]['c'])
    for count in range(0, n_clusters_):
        cluster.append(count)
        tweets.append(len(dbscandf[dbscandf['cluster'] == count]['c']))
        #print(len(dbscandf[dbscandf['cluster'] == count]['c']))
    return cluster, tweets



def cluster(vector,eps,min_samples, metric):
    vector = pairwise_distances(vector,metric = metric)
    model = DBSCAN(eps = eps,min_samples = min_samples).fit(vector)
    return model.labels_


def plot(vector,labels,numLabels):
    for i in range(numLabels):
        plt.scatter(vector[labels==i, 0], vector[labels==i, 1])
    plt.show()

def main():
    print("Pre-processing data...")
    content = prePro()
    vector1 = toVector(content,5)
    print(vector1)

    met = ["cosine", "euclidean"]
    for metri in met:
        print(metri + " Clustering...")
        result = cluster(vector1,3.2,16, metri)


        df = pd.DataFrame(content, columns =['c'])
        #print(df)
        clusterTitles = result
        dbscandf = df
        dbscandf['cluster'] = clusterTitles
        #print(dbscandf['cluster'])
        #print(dict)
        print(dbscandf[dbscandf['cluster'] == 3]['c'])



        print("silhouette_score: " + str(silhouette_score(vector1,result)))
        #print("calinski_harabaz_score: " + str(calinski_harabaz_score(vector, result)))
        #print("davies_bouldin_score: " + str(davies_bouldin_score(vector,result)))
        #diffSet = set(result)
        #print(len(diffSet))

        n_clusters_ = len(set(result)) - (1 if -1 in result else 0)
        n_noise_ = list(result).count(-1)
        print('Cluster numbers：',n_clusters_)
        print('Noise numbers：',n_noise_)

        #cluster histogram
        clus = clusterNum(result, df, n_clusters_)[0]
        tweets = clusterNum(result, df, n_clusters_)[1]
        plt.figure(figsize=(5, 5))
        plt.bar(clus, tweets)
        plt.suptitle('Cluster Distribution')
        plt.xlabel("Cluster#")
        plt.ylabel("Tweets#")
        plt.show()
        time.sleep(5)

       #PCA
        vector = PCA(n_components=2).fit_transform(vector1)
        plot(vector,result,n_clusters_)

if __name__ == '__main__':
    main()