library(tidyverse)
library(vegan)
networkData = read.csv("evonet.csv", header = T)
networkData[is.na(networkData)] = 0

networkData<-networkData%>%
  transform(
            n.links =rowSums(networkData[,c("at","ot","pt","st","vt")]!= 0),
            Gq=2^(diversity(select(networkData, at,ot,pt,st,vt))))

# full model
mod <- lm(n.links~week*donor, data=networkData)

summary(mod)
par(mfrow= c(1,1))
plot(mod) 


# analyse seperately
# just o
od = networkData[networkData$donor == "od",]
mod2 <- lm(Gq~week, data=od)
mod2 <- lm(n.links~week, data=od) #n.links model looks better
summary(mod2)
plot(mod2) 

# just o
pc = networkData[networkData$donor == "pc",]
mod2 <- lm(n.links~week, data=pc)
mod2 <- glm(n.links~week, data=pc, family=poisson)
summary(mod2)
plot(mod2) 

require(nlme)

#try to account for the difference in variance between treatment groups
mod2<-gls(n.links~week,weights=varIdent(form=~1|week),data=pc) #error doesn't seem to work, maybe not enough data

#binomial model 
lost<-cbind(pc$donorLost,pc$donorGFP)
mod.b <- glm(lost~week, data=pc, family=quasibinomial) #overdispersed therefore quasibiomial 

summary(mod.b)
plot(mod.b) #looking good
library(car)
Anova(mod.b)

# without outlier
od = networkData[networkData$donor == "od",]
mod3 <- lm(n.links~week, data=od[od$Gq < 1.3,])
summary(mod3)
plot(mod3) 
