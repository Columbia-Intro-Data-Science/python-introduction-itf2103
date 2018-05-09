from recipe_scrapers import scrape_me

# Remove common filler words that aren't ingredients; I actually ended up keeping some words that could be latent features
import pandas
data = pandas.read_csv("words_remove.csv")
words_remove = data['Words'].tolist()

def clean_ingredients():
    for i in range(len(words_remove)):
        global ingredients
        ingredients = [x.replace(words_remove[i],"") for x in ingredients]
        ingredients = [x.replace("  "," ") for x in ingredients]
        ingredients = [x.strip() for x in ingredients]

        
        
df_recipes = pandas.read_csv("recipe_links.csv")
recipe_links = df_recipes['Link'].tolist()

ingredients_combined = []
titles_list = []
for j in range(len(recipe_links)):
    scrape = scrape_me(recipe_links[j])
    ingredients = scrape.ingredients()
    clean_ingredients()
    ingredients_combined.append(' '.join(ingredients))
    titles_list.append(scrape.title())
    
ingredients_matrix = df_recipes
ingredients_matrix['Title'] = titles_list
ingredients_matrix['Ingredients'] = ingredients_combined

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import linear_kernel

tf = TfidfVectorizer(analyzer='word', ngram_range=(1,3), stop_words='english', binary=True)
tfidf_matrix = tf.fit_transform(ingredients_matrix['Ingredients'])
tfidf_matrix

#cosine similarity
recipe_comparitor = 1
cosine_similarities = linear_kernel(tfidf_matrix[recipe_comparitor], tfidf_matrix).flatten()
cosine_similarities
print('Comparing recipes to: ' + str(ingredients_matrix['Title'][recipe_comparitor]))

cosine_index = cosine_similarities.argsort()[:-12:-1] # Return the 10 best matches for recipes not including recipe used for comparison

similar_items = []
for i in range(len(cosine_index)):
    similar_items.append([(ingredients_matrix['Title'][cosine_index[i]]), cosine_similarities[cosine_index[i]]])
del similar_items[0] # Delete first item from list as that will be the recipe being used for comparison
print('Showing 10 best recipe matches and the cosine similarity')
similar_items