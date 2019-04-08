setwd("D:/Google Drive/Work/SA/Proposals/M&M/Delivery/Refer/Machine learning/Machine learning/ANN")
#Load data into R
concrete  <- read.csv("Concrete_Data.csv", stringsAsFactors = FALSE)

#confirm the particulars of the data
str(concrete)

#check the contents
head(concrete)

#The nine variables in the data frame correspond to the eight features and one outcome we expected, although a problem has become apparent. 
#Neural networks work best when the input data are scaled to a narrow range around zero, and here we see values ranging anywhere from zero

normalize <- function(x) { return((x - min(x)) / (max(x) - min(x))) }

#After executing this code, our normalize() function can be applied to every column in the concrete data frame using the lapply() function as follows: 
concrete_norm <- as.data.frame( lapply( concrete, normalize))

#To confirm that the normalization worked, we can see that the minimum and maximum strength are now 0 and 1
summary( concrete_norm$strength)

#see how the normalized values have a different range than the original values
summary( concrete$strength)

#Any transformation applied to the data prior to training the model will have to be applied 
#in reverse later on in order to convert back to the original units of measurement. 
#To facilitate the rescaling, it is wise to save the original data, or at least the summary statistics of the original data.

#Since the data is already sorted randomly split between test and train directly
concrete_train <- concrete_norm[1:773,]
concrete_test <- concrete_norm[774:1030,]


install.packages("neuralnet")

#include the library
library( neuralnet)

concrete_model <- neuralnet(strength ~ Cement + Slag + Ash + Water + superplastic + coarseagg + fineagg + age, data = concrete_train)

#check the result
plot( concrete_model)


#Checking model performance

#we can use the compute() function to generate predictions on the testing dataset
model_results <- compute(concrete_model,concrete_test[1:8])

#Note that the compute() function works a bit differently from the predict() functions we've used so far. 
#It returns a list with two components: $ neurons, which stores the neurons for each layer in the network, and $ net.results, 
#which stores the predicted values. We'll want the latter:
predicted_strength <- model_results$net.result

cor( predicted_strength, concrete_test $ strength)
#Because the neural network begins with random weights, the predictions can vary from model to model.


#Improving model performance


#As networks with more complex topologies are capable of learning more difficult concepts, let's see what happens when we increase 
#the number of hidden nodes to five. We use the neuralnet() function as before, but add the parameter hidden = 5
concrete_model2 <- neuralnet(strength ~ Cement + Slag + Ash + Water + superplastic + coarseagg + fineagg + age, data = concrete_train, hidden = 5)

#Plotting the network again, we see a drastic increase in the number of connections. How did this impact performance?
plot( concrete_model2)


#Notice that the reported error (measured again by SSE) has been reduced from 6.92 in the previous model to 2.44 here. 
#Additionally, the number of training steps rose from 3222 to 7230, which is no surprise given how much more complex the model has become.

#Applying the same steps to compare the predicted values to the true values, we now obtain a correlation around 0.80, which is a considerable 
#improvement over the previous result
model_results2 <- compute(concrete_model2, concrete_test[1:8])
predicted_strength2 <- model_results2$net.result

cor( predicted_strength2, concrete_test $ strength)
#Has the model improved?