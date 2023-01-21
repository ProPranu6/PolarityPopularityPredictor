# PolarityPopularityPredictor
 > ![Tux, the Linux mascot](/UROPTitlePage.png)
 >
 >
> ## Our Objective
> **“To develop a system, that classifies news into positive and negative or popular and unpopular and recommends news according to the user.“**
>
> ## What do Polarity and Popularity Mean Here
> ### Polarity :
>> Polarity defines the orientation of the expressed sentence.
>>
>> Segregated categories are positive and negative.
>> 
>> We Recommend news with positive vibes in it.
> ### Popularity:
>> Popularity defines the importance of the expressed sentence.
>>
>> Segregated categories are popular and unpopular.
>>
>> News with high popularity is recommended.
>
> ### Tech Stack
>> PYTHON
>>
>> SCIKIT LEARN
>>
>> TENSORFLOW
>>
>> KERAS
>>
>> PANDAS
>>
>> NUMPY
>>
>> MATPLOTLIB
>
> ## Algorithm
>> ### Preprocessing
>> Consider a headline.
>>
>> Tokenization.
>>
>> Stop word removal.
>>
>> Lemmatization.
>>
>> Representation of words into labels named 1- 80,000 numbers.
>
>> ### Polarity Determination
>> Now we proceed to embedding.
>>
>> Word vectors with a dimension of 100 are created.
>>
>> Word vectors are now passed through a Multi-Layered Perceptron.
>>
>> Neural network outputs a value between 0 and 1, 0 being an absolute negative news and 1 being an absolute positive news.
>>
>> This is the Polarity Score.
>
>> ### Popularity Segregation
>> In this particular technique we take the headline of the news article, the news desk, news section, number of comments, keywords,word count, material, date and time >> of publication to judge the popularity of a particular topic in the society.
>>
>> Keywords are converted into categories of 16-length fixed sequences.
>>
>> Box Cox transformation is applied for word count to convert them into a normalised distribution.
>>
>> News Desk, Material and Section will go through Categorical Embedding.
>>
>> Date in DDMMYY and time in HH:MM:SS formats are taken and converted to binary labels indicating articles published before or after evening.
>>
>> This data is driven into a Decision Tree Classifier which outputs either 0 or 1.
>>
>> This is the Popularity Score.
>
>> ### Recomendation Score Calculation
>>> ***RECOMMENDATION SCORE = 0.7(POLARITY SCORE) + 0.3(POPULARITY SCORE)***
>>
>> News with Recommendation Score which is closer to 1 has most possible
>> chance of being recommending whereas score closer to 0 has least possible
>> chance of being recommended. 
> ## Flowchart of the Algorithm
>> ![Tux, the Linux mascot](/UROPTitlePage.png)
>
