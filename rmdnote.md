---
layout: page
title: R Markdown Tutorial
permalink: /rmdnote/
---

R Markdown is a text file format that combines:

- R code for data analysis, and
- text for telling a story / writing a report on your analysis

Think of R Markdown as a way for you to write up your project report and code up your math at the same time, in the same place.

This tutorial explains how to convert R Markdown to PDF. Please go through the following steps:

- Download this [**R Markdown (.Rmd) script**](https://raw.githubusercontent.com/ChicagoBoothML/ML2016/master/code/BostonHousing_KNN_BiasVarTradeOff_CrossValid.Rmd) onto your local disk drive
- Open the downloaded R Markdown script in RStudio
- Make sure the `rmarkdown` package is installed in R. If not, run the command **`install.packages('rmarkdown')`** in RStudio
- In the text-editing window in RStudio, just below the file name, find and press the "**Knit PDF**" button, which will read the content of the script and convert it to a PDF document in the same folder as your R Markdown script file
     - Please note that R Markdown requires [**TeX**](tex) for conversion to PDF
- Inspect the R Markdown script file thoroughly and look at the PDF output to see how the *.Rmd* input is rendered in the final output
