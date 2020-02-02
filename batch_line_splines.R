#!/usr/bin/env Rscript
args = commandArgs(trailingOnly=TRUE)


## ---- echo=FALSE, include=FALSE------------------------------------------
library(mgcv)
library(dplyr)

#setwd('PrintProbLinEx')
#filename = 'anon_blank_blank_00height_amaske-6912_page1r.csv'

directory = file.path(getwd(), args[1])
linesDir = file.path(directory, "lines/")

if (!dir.exists(linesDir)){
  dir.create(linesDir)
}
files = dir(path=directory, pattern='*.csv', full.names=TRUE, recursive=FALSE)
lapply(files, function(filename){

outfile = sub('.csv','_lines.csv',filename)
outfile = file.path(linesDir, basename(outfile))
print(outfile)


dat = read.csv(filename, header = FALSE)
names(dat)='y'
dat$idx = 1:nrow(dat)
alpha = .05
#Subset of data.
datsub = dat[floor(alpha*nrow(dat)):ceiling((1-alpha)*nrow(dat)),]

## ------------------------------------------------------------------------
plot(datsub$y)

## ------------------------------------------------------------------------
lb =20
ub = 140
dlist = lb:ub
devlist = numeric(length(dlist))

## ------------------------------------------------------------------------
for(i in 1:length(dlist)){
  stride = dlist[i]
  datsub$looped = (datsub$idx %% (stride))
  #fit = gam(y~s(looped, bs='cc'), data=datsub, family='gaussian',knots=list(looped=c(0,stride)))
  fit = gam(y~s(looped), data=datsub, family='gaussian')
  devlist[i]=fit$deviance
}
plot(dlist, devlist)
## ------------------------------------------------------------------------
stride = dlist[which.min(devlist)]

## ------------------------------------------------------------------------
dlist = seq(from=stride-2, to=stride+2, length.out=200)
devlist = numeric(length(dlist))
for(i in 1:length(dlist)){
  stride = dlist[i]
  datsub$looped = (datsub$idx %% (stride))
  #fit = gam(y~s(looped, bs='cc'), data=datsub, family='gaussian',knots=list(looped=c(0,stride)))
  fit = gam(y~s(looped), data=datsub, family='gaussian')
  devlist[i]=fit$deviance
}
plot(dlist, devlist)
stride = dlist[which.min(devlist[1:100])]
print(stride)

## ------------------------------------------------------------------------
# which_robust_min = function(x){
#  alpha = 0.1
#  cutoff = min(x) + diff(range(x))*alpha
#  idx = which(x <= cutoff)
#  median(idx)
# }
#Moved in from the fixed version on 6/18/19, seems better for odd cases.
which_robust_min = function(x){
  alpha = 0.1
  cutoff = min(x) + diff(range(x))*alpha
  idx = which(x <= cutoff)
  #Need island handling and wrap around handling.
  #Wrap things around
  idx_wrap = c(idx, idx+length(x))
  #Find largest island
  idx_rle = rle(diff(idx_wrap))
  ii = which.max(idx_rle$lengths)
  aa1 = sum(idx_rle$lengths[1:(ii-1)])+1
  aa2 = sum(idx_rle$lengths[1:ii])+1
  #Use median of that island
  ret = max(1,ceiling(median(idx_wrap[aa1:aa2]) %% length(x)))
  #print(paste(aa1, aa2, idx_wrap[aa1], idx_wrap[aa2], ret))
  ret
}

## ------------------------------------------------------------------------
datsub$looped = (datsub$idx %% (stride))
fit = gam(y~s(looped, bs='cc' ), data=datsub, family='poisson',knots=list(looped=c(0,stride)))

# par(mfrow = c(2,2))
# gam.check((fit))
# summary(fit)
# plot(fit)

dat$looped = (dat$idx%%(stride))
dat$block = (dat$idx - dat$looped)/stride
dat$smoothed = predict(fit, newdata=dat, type='terms')
dat$fitted = predict(fit, newdata=dat, type='response')
#minidx = (dat %>% group_by(block) %>% slice(which.min(smoothed)))$idx
minidx = (dat %>% group_by(block) %>% slice(which_robust_min(smoothed)))$idx


## ---- include=FALSE, echo=FALSE------------------------------------------
#write.table(data.frame(minidx),file='explore_lines3.csv',sep=',',row.names = FALSE, col.names = FALSE)

## ------------------------------------------------------------------------
# cuts = c()
# search_list = -floor(stride/3):floor(stride/3)
# window_width = stride*1 #LAter could take into account whitespace width
# window = -window_width:window_width
# for(i in 1:length(minidx)){
#   idx = minidx[i]
#   if(idx+max(window)>nrow(dat)){next}
#   RSSs = numeric(length(search_list))
#   for(j in 1:length(search_list)){
#     idxs = (idx):(idx+window_width)
#     if(idx+min(search_list)<1){RSSs[j]=Inf; next}
#     RSSs[j] = sum((dat$y[idxs]-dat$fitted[idxs+search_list[j]])^2)#/dat$fitted[idxs])
#   }
#   cuts = c(cuts, minidx[i] - search_list[which.min(RSSs)], minidx[i] - search_list[which.min(RSSs)]+round(stride))
# }
# write.table(data.frame(cuts),file='explore_lines4.csv',sep=',',row.names = FALSE, col.names = FALSE)
# write.table(matrix(cuts, ncol=2, byrow = TRUE),file='explore_lines5.csv',sep=',',row.names = FALSE, col.names = FALSE)

## ------------------------------------------------------------------------

#Heuristic for calculating padding
tmp = exp(dat$smoothed[dat$block==1])
tmp_cutoff = quantile(tmp,0.2)
abline(h=tmp_cutoff)
idx = which(tmp < tmp_cutoff)
if((max(idx) < stride/2) || (min(idx) > stride/2)){
  #One sided
  pad = ceiling((max(idx)-min(idx))/2)
} else {
  #Two sided
  tmpa = max(idx[idx<stride/2])
  tmpb = min(idx[idx>stride/2])
  pad = ceiling(min(tmpb-tmpa, stride - tmpb + tmpa)/2)
}
print(paste('padding',pad))

## ------------------------------------------------------------------------
#pad = pad*4
cuts = c()
rss_store = c()
search_list = -floor(stride/3):floor(stride/3)
window_width = stride*1 #LAter could take into account whitespace width
window = -window_width:window_width
for(i in 1:length(minidx)){
  idx = minidx[i]
  if(idx+max(window)>nrow(dat)){next}
  RSSs = numeric(length(search_list))
  for(j in 1:length(search_list)){
    idxs = (idx):(idx+window_width)
    if(idx+search_list[j]<1){RSSs[j]=Inf; next}
    RSSs[j] = sum((dat$y[idxs+search_list[j]]-dat$fitted[idxs])^2)#/dat$fitted[idxs])
  }
  new_cut = c(minidx[i] + search_list[which.min(RSSs)]-pad, minidx[i] + search_list[which.min(RSSs)]+floor(stride)+pad)
  cuts = c(cuts, new_cut)
  rss_store = c(rss_store, ifelse((min(new_cut) < 1)|(max(new_cut)>nrow(dat)),1e12,min(RSSs)))
}
#was 4 before, now 10.  A bit liberal...
good_cut_idx = which(rss_store < quantile(rss_store, 0.5, na.rm=TRUE)*10)
#write.table(data.frame(cuts),file='explore_lines4.csv',sep=',',row.names = FALSE, col.names = FALSE)
cut_mat = matrix()
write.table(matrix(cuts, ncol=2, byrow = TRUE)[good_cut_idx,],file=outfile,sep=',',row.names = FALSE, col.names = FALSE)

## ------------------------------------------------------------------------


## ------------------------------------------------------------------------
# segs = matrix(cuts, ncol=2, byrow = TRUE)
# par(mfrow=c(5,5))
# for (i in 1:(length(minidx)-1)){
#   plot(dat[minidx[i]:minidx[i+1],]$y)
#   lines(dat[minidx[i]:minidx[i+1],]$fitted, col='red')
#   title(paste(i))
# }
#
# ## ------------------------------------------------------------------------
# segs = matrix(cuts, ncol=2, byrow = TRUE)
# par(mfrow=c(5,5))
# for (i in 1:nrow(segs)){
#   plot(dat[max(0,segs[i,1]):segs[i,2],]$y)
#   lines(dat[max(0,segs[i,1]):segs[i,2],]$fitted, col='red')
#   title(paste(i))
# }
#
# ## ------------------------------------------------------------------------
# #Works pretty well.  Should incorporate a statistical error model that allows different scales
# #Take into account zero inflation from line length.
# par(mfrow=c(1,1))
# segs = matrix(cuts, ncol=2, byrow = TRUE)
# width = segs[2,2]-segs[2,1]+1
# lines = matrix(0,nrow=nrow(segs),ncol=width)
# for (i in 1:nrow(segs)){
#   lines[i,] = dat$y[segs[i,1]:segs[i,2]]
# }
#
# profile = svd(lines)$v[,1]
# profile = profile*sign(mean(profile))
# badness = numeric(nrow(segs))
# badness2 = numeric(nrow(segs))
# for (i in 1:nrow(segs)){
#   # resid = lm(lines[i,]~profile)$residuals
#   # badness[i] = mean(resid^2)
#   fit = glm(lines[i,]~log(profile), family='poisson')
#   badness[i] = fit$deviance
#   badness2[i] = mean((fit$fitted.values-lines[i,])^2/fit$fitted.values)
#   #badness[i] = fit$
# }
# #par(mfrow=c(2,2))
# par(mfrow=c(2,1))
# hist(badness, col='grey', 100)
# hist(badness2, col='grey', 100)
#
# order(badness)
# order(badness2)
#
})

