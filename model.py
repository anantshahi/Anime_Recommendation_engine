import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from flask import Flask, request, jsonify, render_template


animedf = pd.read_csv(r"C:\Users\91847\Documents\GitHub\Anime_Recommendation_engine\Dataset\anime.csv")

ratingsdf = pd.read_csv(r"C:\Users\91847\Documents\GitHub\Anime_Recommendation_engine\Dataset\rating.csv")

anime = animedf.dropna()

count = animedf['rating'].value_counts()

animedf = animedf[animedf['rating'].isin(count[count>=4].index)]


#our datasets after conditional selection 
animedf.rename(columns = {'rating':'animeRating'}, inplace=True)


ratingsdf.rename(columns={'rating':'userRating'}, inplace=True)

combineAnime = pd.merge(ratingsdf,animedf, on ='anime_id')

combineAnime = combineAnime.dropna()

animeRatingCount = (combineAnime.
                     groupby(by=['name'])['animeRating'].
                     count().
                     reset_index().
                     rename(columns = {'animeRating': 'totalRatingCount'})
                     [['name','totalRatingCount']]
                    )


animeRatingCount['name'] = animeRatingCount['name'].apply(lambda x : x.replace('&quot;',''))

finalCombineAnime = combineAnime.merge(animeRatingCount,left_on='name', right_on='name',how='left')

popularity_threshold = 500

rating_popular_anime = finalCombineAnime.query('totalRatingCount>= @popularity_threshold')

rating_popular_anime['name'] = rating_popular_anime['name'].str.lower()

from scipy.sparse import csr_matrix

#We will be dropping duplicates in order to make it efficient from both bookTitle and userID

rating_popular_anime = rating_popular_anime.drop_duplicates(['user_id','name'])

#This setp is important as we will be creating pivot table for the final dataframe 

rating_popular_anime_pivot = rating_popular_anime.pivot(index='name', columns='user_id', values ='animeRating').fillna(0)

rating_popular_anime_matrix = csr_matrix(rating_popular_anime_pivot.values)


#We will be training our data by fitting it into nearest neighbour

from sklearn.neighbors import NearestNeighbors

model_knn  = NearestNeighbors(metric='cosine', algorithm='brute')

model_knn.fit(rating_popular_anime_matrix)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')


def rcm(m):
    
    
    m = m.lower()
    
    if m in rating_popular_anime_pivot.index:
        userquery = m
        distances,indices  = model_knn.kneighbors(rating_popular_anime_pivot.loc[userquery,:].values.reshape(1,-1),
                                         n_neighbors=6)
        animelist = []
        
        for i in range(0, len(distances.flatten())):
            if i == 0:
                print('Recommendations for {0}:\n'.format(userquery))
            else:
               
        
                x = rating_popular_anime_pivot.index[indices.flatten()[i]]
                animelist.append(x)
        return animelist
    


    else:
        return ('Sorry, either it is a typo, or the searched entity is not avaliable, which is rare')
    

@app.route('/recommend',methods=['POST','GET'])

def recommend():
    anime = request.args.get('anime')
    r = rcm(anime)
    if type(r)==type('string'):
        return render_template('recommend.html',anime=anime,r=r,t='s')
    else:
        return render_template('recommend.html',anime=anime,r=r,t='l')
    
    

if __name__ == '__main__':
    app.run(debug=False)
    
    
    

