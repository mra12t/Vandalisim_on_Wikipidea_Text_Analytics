Detecting Vandalism on Wikipedia
================
2022-12-22

### About the Study and the Data Set

Wikipedia is a free online encyclopedia that is open for anyone to
contribute to and edit. However, this can lead to instances of
vandalism, where someone removes or adds inappropriate or promotional
content, or makes subtle changes that alter the meaning of the article.
It can be difficult for humans to catch all instances of vandalism, so
Wikipedia uses bots to revert edits that appear to be vandalism. In this
study, we will try to create a machine learning-based vandalism detector
that can distinguish between valid edits and vandalism. The data for
this task is taken from the revision history of the Language page on
Wikipedia. A script was run to check whether the edits on this page were
kept or reversed, and if they were eventually reversed, they were marked
as vandalism. This may result in some incorrect classifications, but the
script is sufficient for our purposes.

Some basic processing has already been completed on the data, including
lowercasing and removing punctuation. The dataset includes the following
columns:

- Vandal: 1 if the edit was vandalism, 0 if not.

- Minor: 1 if the user marked the edit as “minor,” 0 if not.

- Loggedin: 1 if the user made the edit while logged in to a Wikipedia
  account, 0 if not.

- Added: The unique words added in the edit.  

- Removed: The unique words removed in the edit.

It’s worth noting that the data is not in the traditional bag-of-words
format; instead, it only includes the unique words that were added or
removed in each revision. For example, if a word was removed multiple
times in a single revision, it would only appear once in the “Removed”
column.

### EDA

``` r
wiki = read.csv("wiki.csv", stringsAsFactors = FALSE)
str(wiki)
```

    ## 'data.frame':    3876 obs. of  7 variables:
    ##  $ X.1     : int  1 2 3 4 5 6 7 8 9 10 ...
    ##  $ X       : int  1 2 3 4 5 6 7 8 9 10 ...
    ##  $ Vandal  : int  0 0 0 0 0 0 0 0 0 0 ...
    ##  $ Minor   : int  1 1 0 1 1 0 0 0 1 0 ...
    ##  $ Loggedin: int  1 1 1 0 1 1 1 1 1 0 ...
    ##  $ Added   : chr  "  represent psycholinguisticspsycholinguistics orthographyorthography help text all actions through human ethno"| __truncated__ " website external links" " " " afghanistan used iran mostly that farsiis is countries some xmlspacepreservepersian parts tajikestan region" ...
    ##  $ Removed : chr  " " " talklanguagetalk" " regarded as technologytechnologies human first" "  represent psycholinguisticspsycholinguistics orthographyorthography help all actions through ethnologue relat"| __truncated__ ...

``` r
library(knitr)
wiki$Vandal = as.factor(wiki$Vandal)
x = table(wiki$Vandal)
kable(x)
```

| Var1 | Freq |
|:-----|-----:|
| 0    | 2061 |
| 1    | 1815 |

``` r
barplot(x)
```

![](Detecting%20Vandalism%20on%20Wikipedia_files/figure-gfm/unnamed-chunk-3-1.png)<!-- -->

Our data set has 1815 incident of vandalism.

we will build a model using the bag-of-words approach. We have two
columns of text data, each with its own meaning. For example, adding
rude words has a different connotation than removing rude words. We will
begin by creating a document-term matrix from the Added column. It’s
worth noting that the text has already been lower-cased and stripped of
punctuation.

``` r
library(SnowballC)
library(tm)
```

    ## Loading required package: NLP

``` r
# Creating a corpus for the Added column
corpusAdded = VCorpus(VectorSource(wiki$Added))
# Removing English-Language Stop words
corpusAdded = tm_map(corpusAdded, removeWords, stopwords("english"))
# Stemming the Words
corpusAdded = tm_map(corpusAdded, stemDocument)
# Building the Document Term Matrix (DTM)
dtmAdded = DocumentTermMatrix(corpusAdded)
dtmAdded
```

    ## <<DocumentTermMatrix (documents: 3876, terms: 6675)>>
    ## Non-/sparse entries: 15368/25856932
    ## Sparsity           : 100%
    ## Maximal term length: 784
    ## Weighting          : term frequency (tf)

It appears that our DTM has 6675 terms. So, let’s remove sparse
(infrequent) terms by retaining only those that appear in at least 0.3%
of the revisions.

``` r
sparseAdded = removeSparseTerms(dtmAdded, 0.997)
sparseAdded
```

    ## <<DocumentTermMatrix (documents: 3876, terms: 166)>>
    ## Non-/sparse entries: 2681/640735
    ## Sparsity           : 100%
    ## Maximal term length: 28
    ## Weighting          : term frequency (tf)

The terms are now 166!. Let’s convert it to a data frame now and prepend
all the terms with “A”.

``` r
#Converting sparseAdded into a Data Frame
wordsAdded = as.data.frame(as.matrix(sparseAdded))
#Prepending all the words with "A"
colnames(wordsAdded) = paste("A", colnames(wordsAdded))
```

Now Let’s do the same for the removed column.

``` r
# Creating a corpus for the Removed column
corpusRemoved = VCorpus(VectorSource(wiki$Removed))
# Removing English-Language Stop words
corpusRemoved = tm_map(corpusRemoved, removeWords, stopwords("english"))
# Stemming the Words
corpusRemoved = tm_map(corpusRemoved, stemDocument)
# Building the Document Term Matrix (DTM)
dtmRemoved = DocumentTermMatrix(corpusRemoved)
dtmRemoved
```

    ## <<DocumentTermMatrix (documents: 3876, terms: 5403)>>
    ## Non-/sparse entries: 13293/20928735
    ## Sparsity           : 100%
    ## Maximal term length: 784
    ## Weighting          : term frequency (tf)

``` r
sparseRemoved = removeSparseTerms(dtmRemoved, 0.997)
sparseRemoved
```

    ## <<DocumentTermMatrix (documents: 3876, terms: 162)>>
    ## Non-/sparse entries: 2552/625360
    ## Sparsity           : 100%
    ## Maximal term length: 28
    ## Weighting          : term frequency (tf)

``` r
wordsRemoved = as.data.frame(as.matrix(sparseRemoved))
colnames(wordsRemoved) = paste("R", colnames(wordsRemoved))
```

``` r
ncol(wordsAdded)
```

    ## [1] 166

``` r
ncol(wordsRemoved)
```

    ## [1] 162

Now let’s combine the two new data sets.

``` r
wikiWords = cbind(wordsAdded,wordsRemoved)
```

Let’s add the Vandal column from the original data set.

``` r
wikiWords$Vandal = wiki$Vandal
```

``` r
ncol(wikiWords)
```

    ## [1] 329

Now that the data set is processed and ready, we can split it.

``` r
library(caTools)
set.seed(123)
spl = sample.split(wikiWords$Vandal, 0.7)
train = subset(wikiWords, spl == TRUE)
test = subset(wikiWords, spl == FALSE)
```

Let’s see what a base line model looks like!

### Base Line Model

``` r
x = table(test$Vandal)
kable(x)
```

| Var1 | Freq |
|:-----|-----:|
| 0    |  618 |
| 1    |  545 |

A base line model of the testing data set would predict Not Vandal all
the time and have an accuracy of :

``` r
x[1]/sum(x)
```

    ##         0 
    ## 0.5313844

53% of the time.

### Building a CART Model

``` r
library(rpart)
library(rpart.plot)
wikiCART = rpart(Vandal ~ ., data = train, method = "class")
prp(wikiCART)
```

![](Detecting%20Vandalism%20on%20Wikipedia_files/figure-gfm/unnamed-chunk-15-1.png)<!-- -->

``` r
predCART = predict(wikiCART, newdata = test, type = "class")
x = table(test$Vandal, predCART)
kable(x)
```

|     |   0 |   1 |
|:----|----:|----:|
| 0   | 618 |   0 |
| 1   | 533 |  12 |

``` r
#Accuracy
(x[1]+x[4])/sum(x)
```

    ## [1] 0.5417025

The accuracy of the model is 54%, barely beating the base line model.
There is no evidence that there was an issue with the split. The CART
model did not over fit, as indicated by the accuracy of the model on the
training set. It is possible that the sparsity parameter was set too
high, leading to over-sparsification, but this seems unlikely. The most
likely explanation is that the bag-of-words approach simply did not
perform well in this case.

One class of words that we will focus on is website addresses, also
known as URLs (Uniform Resource Locators). These are made up of two main
parts, such as “<http://www.google.com>,” with the first part being the
protocol (usually “http” for HyperText Transfer Protocol) and the second
part being the site address (“www.google.com”). In the data, website
links have been stripped of punctuation and appear as one word, like
“httpwwwgooglecom.” Let’s assume that the inclusion of web addresses,
particularly those for promotional or irrelevant sites, may be a sign of
vandalism. To search for the presence of web addresses in the Added
column, we can use the grepl function to check for the presence of
“http.” This function returns TRUE if the string is found within another
string.

### Building a possibly better CART Model

``` r
#Creating a copy of the data set
wikiWords2 = wikiWords
wikiWords2$HTTP = ifelse(grepl("http", wiki$Added, fixed = TRUE), 1, 0)
x = table(wikiWords2$HTTP)
kable(x)
```

| Var1 | Freq |
|:-----|-----:|
| 0    | 3659 |
| 1    |  217 |

So we found that 217 added words had HTTP.

``` r
train2 = subset(wikiWords2, spl == TRUE)
test2 = subset(wikiWords2, spl == FALSE)
```

``` r
wikiCART2 = rpart(Vandal ~ ., data = train2, method = "class")
prp(wikiCART2)
```

![](Detecting%20Vandalism%20on%20Wikipedia_files/figure-gfm/unnamed-chunk-20-1.png)<!-- -->

Let’s see the accuracy of the new model.

``` r
predCART2 = predict(wikiCART2, newdata = test2, type = "class")
x = table(test2$Vandal, predCART2)
kable(x)
```

|     |   0 |   1 |
|:----|----:|----:|
| 0   | 609 |   9 |
| 1   | 488 |  57 |

``` r
(x[1]+x[4])/sum(x)
```

    ## [1] 0.5726569

The accuracy of this CART model is 57%, an increase of 3% from the
previous CART model.

Another approach we can use that could possibly improve the model is
using the count of the words that are added and removed to predict the
outcome. In fact, the count of the words might be more significant than
the words themselves. This count does exist in the DTM that we already
constructed, so let’s utilize it.

### Another Possible Improvment

``` r
wikiWords2$NumWordsAdded = rowSums(as.matrix(dtmAdded))
wikiWords2$NumWordsRemoved = rowSums(as.matrix(dtmRemoved))
mean(wikiWords2$NumWordsAdded)
```

    ## [1] 4.050052

``` r
mean(wikiWords2$NumWordsRemoved)
```

    ## [1] 3.5129

``` r
train3 = subset(wikiWords2, spl == TRUE)
test3 = subset(wikiWords2, spl == FALSE)
```

``` r
wikiCART3 = rpart(Vandal ~ ., data = train3, method = "class")
prp(wikiCART3)
```

![](Detecting%20Vandalism%20on%20Wikipedia_files/figure-gfm/unnamed-chunk-25-1.png)<!-- -->

``` r
predCART3 = predict(wikiCART3, newdata = test3, type = "class")
x = table(test3$Vandal, predCART3)
kable(x)
```

|     |   0 |   1 |
|:----|----:|----:|
| 0   | 514 | 104 |
| 1   | 297 | 248 |

``` r
#Accuracy
(x[1]+x[4])/sum(x)
```

    ## [1] 0.6552021

65%! this is a great increase in the accuracy! Now, there’s one more
improvement that we can made which is to use the meta data, the other
variables that we had, but we didn’t use.

### Building the Final CART Model

``` r
wikiWords3 = wikiWords2
wikiWords3$Minor = wiki$Minor
wikiWords3$Loggedin = wiki$Loggedin
```

``` r
train4 = subset(wikiWords3, spl == TRUE)
test4 = subset(wikiWords3, spl == FALSE)
```

``` r
wikiCART4 = rpart(Vandal ~ ., data = train4, method = "class")
prp(wikiCART4)
```

![](Detecting%20Vandalism%20on%20Wikipedia_files/figure-gfm/unnamed-chunk-30-1.png)<!-- -->

``` r
predCART4 = predict(wikiCART4, newdata = test4, type = "class")
x = table(test4$Vandal, predCART4)
kable(x)
```

|     |   0 |   1 |
|:----|----:|----:|
| 0   | 595 |  23 |
| 1   | 304 | 241 |

``` r
#Accuracy 
(x[1]+x[4])/sum(x)
```

    ## [1] 0.7188306

Excellent! The accuracy of the final model is almost 72%! This is a
great increase and a great improvement from all the previous CART models
and the base line model!
