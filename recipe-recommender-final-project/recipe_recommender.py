from flask import Flask, jsonify, render_template, request
app = Flask(__name__)

import pandas
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/', methods=['POST'])
def recommend():
    input = request.form['text']

    #from sqlalchemy import create_engine
    #import psycopg2
    #try:
    #    conn = psycopg2.connect("dbname='recipe_recommender' user='postgres' host='localhost' password='recipe' port='5432'")
    #except:
    #    print('Unable to connect to database')
    #engine = create_engine('postgresql://postgres:recipe@localhost:5432/recipe_recommender')
    #ingredients_matrix = pandas.read_sql_table('Ingredients Matrix', engine)


    ingredients_matrix = pandas.read_csv('ingredients_matrix.csv')
    df_input = pandas.DataFrame({'Link':['N/A'],'Title':['N/A'],'Ingredients':[input]})
    df_input = df_input[['Link','Title','Ingredients']]
    df_ingredients_merge = pandas.concat([df_input, ingredients_matrix])

    tf = TfidfVectorizer(analyzer='word', ngram_range=(1,3), stop_words='english', binary=True)
    tfidf_matrix = tf.fit_transform(df_ingredients_merge['Ingredients'])

    #cosine similarity
    recipe_comparitor = 0
    cosine_similarities = linear_kernel(tfidf_matrix[recipe_comparitor], tfidf_matrix).flatten()
    cosine_similarities
    #print('Comparing recipes to: ' + str(ingredients_matrix['Title'][recipe_comparitor]))

    cosine_index = cosine_similarities.argsort()[:-12:-1] # Return the 10 best matches for recipes not including recipe used for comparison

    similar_items = []
    for i in range(len(cosine_index)):
        similar_items.append([(ingredients_matrix['Title'][cosine_index[i]]), cosine_similarities[cosine_index[i]]])
    del similar_items[0] # Delete first item from list as that will be the recipe being used for comparison
    return jsonify('Showing 10 best recipe matches and the cosine similarity', similar_items)

if __name__ == '__main__':
    app.run(debug=True)