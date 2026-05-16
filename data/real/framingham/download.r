install.packages("riskCommunicator")

library(riskCommunicator)

data("framingham", package = "riskCommunicator")

write.csv(framingham, "framingham.csv", row.names = FALSE)
