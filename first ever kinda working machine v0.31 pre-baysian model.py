##################################
###Sentiment Analysis BOW model###
##################################

import os
import datetime
import pandas as pd
import csv
import re

#####
##1## extract the documents in a positive and negative list
#####

print("start")

##### The first thing to do is to go ahead and extract all the documents for a given rating score from the files and
##### list them then put them in a list of lists by rating:

positive_documents_by_rating = []

for i in range (7, 11):

    doc_string = ""

    for filename in os.listdir ("data\\aclImdb\\train\\pos\\"):

        if filename.endswith ("_" + str (
                i) + ".txt"):  # here we are detecting the files with the relevant rating which is found at the end of the file name

            document = open ("data\\aclImdb\\train\\pos\\" + filename, "r",
                             encoding="utf8")  # open the document if it has a relevant filename

            for line in document:  # generate the document as a string
                doc_string += line.lower () + " "

            document.close ()

    positive_documents_by_rating.append (doc_string)

print("positive docs assigned")

negative_documents_by_rating = []

for i in range (1, 5):

    doc_string = ""

    for filename in os.listdir ("data\\aclImdb\\train\\neg\\"):

        if filename.endswith ("_" + str (
                i) + ".txt"):  # here we are detecting the files with the relevant rating which is found at the end of the file name

            document = open ("data\\aclImdb\\train\\neg\\" + filename, "r",
                             encoding="utf8")  # open the document if it has a relevant filename

            for line in document:  # generate the document as a string
                doc_string += line.lower () + " "

            document.close ()

    negative_documents_by_rating.append (doc_string)

print ("the length of negative docs by rating is: ", len (negative_documents_by_rating[2]))


# positive documents by rating is now a list of 4 strings each corresponding to a rating

# negative documents by rating is similar except that the ratings go 1,2,3,4 instead of 7,8,9,10



#####
##2## Tokenise the words/symbol groups
#####

#####we need to replace all non-letter characters with spaces to make them in to words and account for some typos:


def key_val_mapping_to_csv1 (output_file_name, mapping):

    v = list (mapping.values ())
    k = list (mapping.keys ())  #concat these lists

    z = []
    for i in range (len (k)):
       z.append (k[i] + str(v[i]))

    with open (output_file_name, 'w') as csvfile:
        csv_writer = csv.writer (csvfile, delimiter=',',
                                 quotechar='|', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow (z)


        #k = list (mapping.keys ())

        #z = []
        #for i in range (len (k)):
        #    z.append (k[i] + ", " + ",".join([str(j) for j in v[i]]) + "\n")

        #print("ZED VALUES: " , z[:10])

        #csv_writer.writerow ([zed.encode ('utf-8') for zed in z])



def key_val_mapping_to_csv (output_file_name, mapping):
    with open(output_file_name, 'w') as outfile:

        values_in_mapping = list (mapping.values ())
        keys_in_mapping = list (mapping.keys ())

        value_map_vector = []
        for i in range (len (keys_in_mapping)):
            value_map_vector.append (keys_in_mapping[i] + ", " + ", ".join([str(e) for e in values_in_mapping[i]]) + "\n")

        print("VALUE MAP VECTOR:", value_map_vector[:10])

        for line in value_map_vector:

            try:
                outfile.write (line)
            except:
                continue


def expand_around_chars(text, characters):
    for char in characters:
        text = text.replace (char, " " + char + " ")
    return text


# now we need to go through each string and actually implement the replacement:

for i in range (len (positive_documents_by_rating)):
    positive_documents_by_rating[i] = expand_around_chars (positive_documents_by_rating[i], '";.,()[]{}:;?/<>')

for i in range (len (negative_documents_by_rating)):
    negative_documents_by_rating[i] = expand_around_chars (negative_documents_by_rating[i], '";.,()[]{}:;?/<>')


#####
##3## Create a key-val mapping which counts up the occurrences of each word in each rating category and normalise the vectors
#####

##### set up the dataframe



def word_tally1(documents, ratings_index):
    global ratings_map
    global wordlist

    hell_count = 0

    for i in range (len (documents)):

        split_document = documents[i].split ()

        for word in split_document:

            if word == "yes":
                hell_count += 1

            if word[0].isalpha ():
                try:
                    ratings_map[word][ratings_index[i]] += 1
                except:

                    ratings_map[word] = [0, 0, 0, 0, 0, 0, 0, 0]
                    ratings_map[word][ratings_index[i]] += 1

    print ("the hellcount is " + str (hell_count))



y = datetime.datetime.now ()


ratings_index = [0, 1, 2, 3]
word_tally1 (negative_documents_by_rating, ratings_index)
ratings_index = [4, 5, 6, 7]
word_tally1 (positive_documents_by_rating, ratings_index)

x = datetime.datetime.now ()



print ("\n\n")


key_val_mapping_to_csv ("pre-normalisation.csv", ratings_map)

#####now we have a mapping which contains all the words and their scores. we need to normalise the vectors to one. We
#####can also remove the rows which have a count of less than 3 or something

print ("the number of unique words BEFORE normalisation is: ", len (ratings_map))

i = 0
for word in ratings_map:
    i += 1
    print (str (i) + " " + word)
    if i == 10:
        break

print ("\n\n")

# NORMALISATION LOOP:

for key in list(ratings_map.keys()):

    row_sum = 0

    for j in range (8):
        row_sum += ratings_map[key][j]

    if row_sum > 1:
        for j in range (8):
            ratings_map[key][j] = ratings_map[key][j] / row_sum
    else:
        del ratings_map[key]

x = datetime.datetime.now ()
# print ("the time to normalise the vectors is: ", x - y)

print ("the number of unique words AFTER normalisation is: ", len (ratings_map))

key_val_mapping_to_csv ("post-normalisation.csv", ratings_map)

i = 0


#####
##3## now we need to make a wordscore vector
#####

##### we need to create a new vector for the word scores, this will be side by side with the word vector which we can
##### concatenate after. to do this we need to take the row index (indexed by the word vector), then go through
##### multiplying by the score (then divide by 10 or no???)


wordscore = {}

for word in ratings_map:

    expected_value = 0

    for i in range (4):
        expected_value += ratings_map[word][i] * (i + 1)
    for i in range (4,8):
        expected_value += ratings_map[word][i] * (i + 3)

    wordscore[word] = expected_value

#key_val_mapping_to_csv1 ("wordscore.csv", wordscore)

##### now we should be able to "score" a document:

#chuck a while True in here to loop constantly

while True:

    test_document = input ("write something: ")

    split_document = test_document.split ()
    print(split_document)

    print("you wrote " + str (len (split_document)) + " words.....")

    document_score = 0
    j = 0

    for word in split_document:
        try:
            document_score += wordscore[word]
        except:
            print("unfortunately the word '" + word +  "' is not in the list, so it has been left out, maybe you did not spell it correctly...")
            j += 1

    try:
        document_score = document_score / (len (split_document) - j)
        print ("with a score of: " + str (document_score))
    except:
        print("something has gone wrong sorry")





