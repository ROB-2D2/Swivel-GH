###
### This one does the dictionary replacement thing, that is basically all, no bigrams
###

from tkinter import *
from tkinter import ttk
import os
from nltk.corpus import stopwords

###
#1# Define the functions
###

def key_val_mapping_to_csv(output_file_name, mapping):
    with open (output_file_name, 'w') as outfile:

        values_in_mapping = list (mapping.values ())
        keys_in_mapping = list (mapping.keys ())

        value_map_vector = []
        for i in range (len (keys_in_mapping)):
            if keys_in_mapping[i] == 'danning':
                print (keys_in_mapping[i] + ", " + ", ".join ([str (e) for e in values_in_mapping[i]]) + "\n")

            value_map_vector.append (keys_in_mapping[i] + ", " + ", ".join ([str (e) for e in values_in_mapping[i]]) + "\n")

        print ("VALUE MAP VECTOR:", value_map_vector[:10])

        for line in value_map_vector:

            try:
                outfile.write (line)
            except:
                continue

def expand_around_chars(text, characters):
    for char in characters:
        text = text.replace (char, " " + char + " ")
    return text

def word_tally1(documents, ratings_index):
    global ratings_map
    global wordlist

    hell_count = 0

    for i in range (len (documents)):

        split_document = documents[i]#.split ()

        tokenized_docs_no_stopwords = []

        for word in split_document:
            if not word in stopwords.words ('english'):
                tokenized_docs_no_stopwords.append (word)

        for word in tokenized_docs_no_stopwords:

            if word[0].isalpha ():
                try:
                    ratings_map[word][ratings_index[i]] += 1
                except:
                    ratings_map[word] = [0, 0, 0, 0, 0, 0, 0, 0]
                    ratings_map[word][ratings_index[i]] += 1

    print ("the hellcount is " + str (hell_count))

def dictionary_list_creator(dictionary_file_name, first_letter):    #this function makes a list which contains all words in the dictionary which begin with the letter passed in to the second argument

    letter_dictionary = []

    with open ("C:\\Users\\dripd\\Google Drive\\Robs Docs\\Swivel\\English Word Dictionary\\" + dictionary_file_name, "r") as readfile:
        for line in readfile:
            if line[0] == first_letter:
                letter_dictionary.append(line)

    for i in range(len(letter_dictionary)):         #here we strip whitespaces, I am not entirely sure why to be honest, maybe remove??? shouldnt break anything
        letter_dictionary[i] = letter_dictionary[i].strip()

    return(letter_dictionary)

def minimumEditDistance (s1, s2):        #this function comes from Rosetta code most efficient calc of levenshtein dist
    if len (s1) > len (s2):
        s1, s2 = s2, s1
    distances = range (len (s1) + 1)
    for index2, char2 in enumerate (s2):
        newDistances = [index2 + 1]
        for index1, char1 in enumerate (s1):
            if char1 == char2:
                newDistances.append (distances[index1])
            else:
                newDistances.append (1 + min ((distances[index1],
                                               distances[index1 + 1],
                                               newDistances[-1])))
        distances = newDistances
    return distances[-1]

def LCS_consonant_calculator (string1, string2):

    string1 = ''.join ([letter for letter in string1 if letter not in "aeiou"])     #strips the string of vowels
    string2 = ''.join ([letter for letter in string2 if letter not in "aeiou"])

    start_point = 0             #this is so that we can iterate from the corect point in the j loop (dont want to recount letter matching)
    LCS_length = 0

    for i in range (len (string1)):

        LCS_length_has_changed = False

        for j in range(start_point, len (string2)):

            if string1[i] == string2[j] and not LCS_length_has_changed:

                LCS_length += 1         #actually need to break the loop here
                LCS_length_has_changed = True
                start_point = j + 1

    return LCS_length

#this should be faster than the "other" method apparently. Maybe compare the 2 for speed
def keywithmaxval(mapping):

    v = list (mapping.values ())
    k = list (mapping.keys ())
    return k[v.index (max (v))]

def determine_and_return_most_similar_words(tokenised_sentence, dictionary, threshold_value, dictionary_set):       #this basically needs to be tweaked to return an edited list

    for i in range(len(tokenised_sentence)):

        if tokenised_sentence[i] not in dictionary_set and tokenised_sentence[i][0].isalpha ():

            percentage_process_value['text'] = str(i/len(tokenised_sentence)*100)

            map = {}

            try:
                alphabet_index = "abcdefghijklmnopqrstuvwxyz".index(tokenised_sentence[i][0])   #find the word's alphabetical index (the position in the dictionary where a = 1, b = 2, etc..)
            except:
                continue

            for word in dictionary[alphabet_index]:

                LCS_ratio = LCS_consonant_calculator (tokenised_sentence[i], word) / len(word)  #find the LCS ratio of the word in the sentence against the word in the dictionary
                edit_distance = minimumEditDistance (tokenised_sentence[i], word)               #do the same for the edit distance

                if (LCS_ratio / (edit_distance)) > threshold_value:              #if this ratio exceeds the threshold value for that word, add the ratio as a value to the mapping with the key being the corresponding word

                    #print(word)

                    #print("The similarity measure is: ",(LCS_ratio / (edit_distance + 0.0000000001)))

                    map [word] = LCS_ratio / (edit_distance)
            try:
                tokenised_sentence[i] = keywithmaxval(map)                                      #after we have done the above for every word change the word in the sentence to the word in the mapping which has the highest similarity measure and then move on to the next word in the sentence
            except:
                continue

    return tokenised_sentence

### c) Put the words in each segment through the similarity measure and pick up the best one
def execute_calculation(*args):
    ###
    #2# Open up the documents, put the sentences through the similarity comparison code
    ###

    ### a) Open the docs
    # After this the documents are just one long string in the lists defined in the next 2 lines

    positive_documents_by_rating = []
    negative_documents_by_rating = []
    document_count_by_class = [0, 0, 0, 0, 0, 0, 0, 0]

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
                document_count_by_class[i - 3] += 1

        positive_documents_by_rating.append (doc_string)

    print ("positive docs assigned")

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
                document_count_by_class[i - 1] += 1

        negative_documents_by_rating.append (doc_string)

    print ("negative docs assigned: ", len (negative_documents_by_rating))

    print (len (negative_documents_by_rating[0].split ()))

    class_probabilities = []
    total_number_of_docs = sum (document_count_by_class)

    for i in range (8):
        class_probabilities.append (document_count_by_class[i] / total_number_of_docs)

    print ("CLASS PROBABILITIES: ", class_probabilities)

    print("expanding around chars")

    for i in range (len (positive_documents_by_rating)):
        positive_documents_by_rating[i] = expand_around_chars (positive_documents_by_rating[i], '";.,()[]{}:;?/<>\'!')

    for i in range (len (negative_documents_by_rating)):
        negative_documents_by_rating[i] = expand_around_chars (negative_documents_by_rating[i], '";.,()[]{}:;?/<>\'!')

    print ("FINISHED expanding around chars")

    ### b) load up the dictionary

    file = "google 10000 lined.txt"

    dictionary = []  # these 3 lines will create a list of lists, where each of the lists in the list contain the words corresponding to that letter in the dictionary
    for letter in "abcdefghijklmnopqrstuvwxyz":
        dictionary.append (dictionary_list_creator (file, letter))

    dictionary_set = set ()
    for list1 in dictionary:
        for item in list1:
            dictionary_set.add (item)

    for list1 in dictionary:
        print ("the length of the elements in the dictionary are: ", len (list1))

    print ("dictionary and dictionary set generated")

    for j in range (len (negative_documents_by_rating)):
        print("WORK; " + str (j + 1))
        current_rating_value['text'] = str (j + 1)
        negative_documents_by_rating[j] = determine_and_return_most_similar_words (negative_documents_by_rating[j].split (), dictionary, 0.3, dictionary_set)

    for j in range (len (positive_documents_by_rating)):
        current_rating_value['text'] = str (j + 7)
        print ("WORK; " + str (j + 7))
        positive_documents_by_rating[j] = determine_and_return_most_similar_words (positive_documents_by_rating[j].split (), dictionary, 0.3, dictionary_set)

    ### d) do the replacement and count up

    print (negative_documents_by_rating[1][:10])

    print ("Start making the dataframe....")
    global ratings_map
    ratings_map = {"hello": [0, 0, 0, 0, 0, 0, 0, 0]}

    ratings_index = [0, 1, 2, 3]
    word_tally1 (negative_documents_by_rating, ratings_index)       #SOMETHING IS BROKEN HERE AND IS SHIFTTING THE COUNT ONE CELL TO THE RIGHT
    ratings_index = [4, 5, 6, 7]
    word_tally1 (positive_documents_by_rating, ratings_index)

    key_val_mapping_to_csv ("pre-normalisation.csv", ratings_map)
    print("DONE! pre-normalisation.csv")

    ###in order to compute the class conditional probabilities we will need to know the number of words in each class:

    words_in_class = {}

    for i in range (1, 11):

        if (0 < i < 5):

            words_in_class[i] = len (positive_documents_by_rating[i - 1])

        elif (6 < i < 11):

            words_in_class[i] = len (negative_documents_by_rating[i - 7])

    # this is just to check:

    print ("the number of words in each class is:")

    for i in range (1, 11):
        try:
            print (words_in_class[i])
        except:
            continue

    print ("\n\n")

    for key in list(ratings_map.keys ()):

        row_sum = 0

        for j in range (8):
            row_sum += ratings_map[key][
                j]  # we are just counting up how many total appearances of the word there were here

        i = 1
        if row_sum > 3:  # here we should change the number if we want only words that appear more than that number of times (i have somewhat arbitrarily chosen 3)
            for j in range (8):
                ratings_map[key][j] = ratings_map[key][j] / words_in_class[i]  # this is the calculation that tells us the probability of a word appearing in a class, in other words what is the probability of a word in a given class being exactly this word, the number of appearances of the word divided by the total number of words that appear in the class
                if i == 4:
                    i += 3
                else:
                    i += 1
        else:
            del ratings_map[key]

    key_val_mapping_to_csv ("class conditional probabilities for all words.csv", ratings_map)

    print ("normalised version of the 'wonderful' vector...:", ratings_map["wonderful"])


##############################
##############################  TKINTER SECTION:
##############################
###open the window and name it
root = Tk ()
root.title ("Calculation Progress Tracker")

###set up the features of the window
mainframe = ttk.Frame (root, padding="3 3 12 12")
mainframe.grid (column=0, row=0, sticky=(N, W, E, S))
mainframe.columnconfigure (0, weight=1)         #this just tells the contents of the window to expand if the window is expanded
mainframe.rowconfigure (0, weight=1)

###set some global variables which will be updated as we track progress
# percent_through_rating_category = StringVar ()
# current_rating_category = StringVar ()

###some static text lables, intended to lable the segments below
current_rating_lable = ttk.Label (mainframe, text="Current Rating Category")
current_rating_lable.grid (column=1, row=1, sticky=W)
percentage_process_lable = ttk.Label (mainframe, text="Percentage Progress")
percentage_process_lable.grid (column=2, row=1, sticky=E)

###the place where we will indicate how far through the calculation we have progressed, under the lables above:
current_rating_value = ttk.Label (mainframe, text="NOT YET STARTED")
current_rating_value.grid (column=1, row=2, sticky=(N, W))
percentage_process_value = ttk.Label (mainframe, text="0%")
percentage_process_value.grid (column=2, row=2, sticky=(N, W))

#create a start button which calls the caculate function
ttk.Button (mainframe, text="Calculate", command=execute_calculation).grid (column=3, row=3, sticky=W)

###put some padding around the widgets within the frame for appearance so they dont get scrunched together:
for child in mainframe.winfo_children (): child.grid_configure (padx=5, pady=5)

root.bind ('<Return>', execute_calculation)

current_rating_value['text'] = str (0)

# def change_rating_number():
#     current_rating_value['text'] = "NONE"
#     current_rating_value['text'] = "NONE again"
#
#
# ttk.Button (mainframe, text="Change the Value", command=change_rating_number).grid (column=3, row=4, sticky=W)

root.mainloop ()

##############################
############################## END OF TKINTER SECTION:
##############################

