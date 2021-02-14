# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 03:20:35 2021

@author: STUFFBOX
"""
#import required libraries
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    #to give a heading
    html_temp = """
    <div style="background-color:blue;padding:10px">
    <h2 style="color:white;text-align:center;">Streamlit Loan defaulter classification</h2>
    </div>
    
    <p>This data is from LendingClub.com. Lending Club connects people who need money (borrowers) with people who have money (investors). Hopefully, as an investor you would want to invest in people who showed a profile of having a high probability of paying you back. We will try to create a model that will help predict this.</p>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    
    #loading the loans data using pandas
    st.title("Data Loading")
    loans = pd.read_csv('loan_data.csv')
    if st.checkbox("Lets have a look at the data"):
        st.write(loans.head())
    if st.checkbox("Check number of rows and columns"):
        st.write(loans.shape)
    if st.checkbox("Check data columns and their description"):
        html_temp1 = """
   
    <p><b>credit.policy</b>: 1 if the customer meets the credit underwriting criteria of LendingClub.com, and 0 otherwise.
    \n<b>purpose</b>: The purpose of the loan (takes values "credit_card", "debt_consolidation", "educational", "major_purchase", "small_business", and "all_other").
    \n<b>int.rate</b>: The interest rate of the loan, as a proportion (a rate of 11% would be stored as 0.11). Borrowers judged by LendingClub.com to be more risky are assigned higher interest rates.
    \n<b>installment</b>: The monthly installments owed by the borrower if the loan is funded.
    \n<b>log.annual.inc</b>: The natural log of the self-reported annual income of the borrower.
    \n<b>dti</b>: The debt-to-income ratio of the borrower (amount of debt divided by annual income).
    \n<b>fico</b>: The FICO credit score of the borrower.
    \n<b>days.with.cr.line</b>: The number of days the borrower has had a credit line.
    \n<b>revol.bal</b>: The borrower's revolving balance (amount unpaid at the end of the credit card billing cycle).
    \n<b>revol.util</b>: The borrower's revolving line utilization rate (the amount of the credit line used relative to total credit available).
    \n<b>inq.last.6mths</b>: The borrower's number of inquiries by creditors in the last 6 months.
    \n<b>delinq.2yrs</b>: The number of times the borrower had been 30+ days past due on a payment in the past 2 years.
    \n<b>pub.rec</b>: The borrower's number of derogatory public records (bankruptcy filings, tax liens, or judgments).
    \n<b>not.fully.paid</b>: Target column ,0 If the loan is paid fully, 1 otherwise </p>
    """
        st.markdown(html_temp1,unsafe_allow_html=True)
    st.title("Checking Some EDA")   
    st.write("Checking what the existing data says,plotting paying capability(paid or not paid according to FICO score.")
    
    fig, ax = plt.subplots()
    ax.hist(loans[loans['not.fully.paid']==1]['fico'], bins=20,alpha=0.5,color='blue',label='not.fully.paid=1')
    ax.hist(loans[loans['not.fully.paid']==0]['fico'], bins=30,alpha=0.5,color="red",label='not.fully.paid=0')
    plt.legend()
    plt.xlabel('FICO')
    st.pyplot(fig)
    st.write("What is the purpose of taking loans?")
    plt.figure(figsize=(11,15))
    fig1, ax = plt.subplots()
    sns.countplot(x='purpose',hue='not.fully.paid',data=loans,palette='Set1')
    ax.set_xticklabels(ax.get_xticklabels(),fontsize=8,rotation=60)
    st.pyplot(fig1)
    st.title("Preparing Data")
    st.write("Data has a categorical column,which needs to be converted into numeric")
    cat_feats = ['purpose']
    final_data = pd.get_dummies(loans,columns=cat_feats,drop_first=True)
    if st.checkbox("Lets have a look at the new data after conversion to all numeric"):
        st.write(final_data.head())
    
    st.title("Modelling and prediction")
    from sklearn.model_selection import train_test_split
    
    X = final_data.drop('not.fully.paid',axis=1)
    y = final_data['not.fully.paid']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)
    
    st.write("Here, Decision tree classifier have been used for predicting the class ")
    from sklearn.tree import DecisionTreeClassifier
    st.write("Choose the min sample split to be made in decision tree algorithm")
    m=st.slider("min_samples_split",2,6)
    dtree = DecisionTreeClassifier(min_samples_split=m)
    dtree.fit(X_train,y_train)
    
    if st.checkbox("Check Accuracy Score of the model"):
        st.write(dtree.score(X_test,y_test))
    
    st.write("Given a set of feature of one borrower, lets predict if loan will be fully paid or not.")
    st.write(X.head(1))
    result=""
    if st.button("Predict"):
        result=dtree.predict([[1,	0.1189,	829.10	,11.350407	,19.48,	737,	5639.958333,	28854,	52.1,	0,	0,	0,	0,	1,	0,	0,	0,	0]])
        if result==0:
            st.success('There are high chances that the loan will be fully paid by the borrower.')
        else:
            st.success('Oops! the loan may not be fully paid by this borrower, re-think before investing!!')
    
   
    
    

if __name__=='__main__':
    main()
