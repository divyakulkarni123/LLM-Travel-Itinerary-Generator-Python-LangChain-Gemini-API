from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from google.colab import userdata

name= input("Enter your name: ")
city= input(f"Hey {name}, Enter your city: ")
number_of_days= input("How many days you would travel: ")
travel_style= input("What is your travel style (relaxed/busy): ")
interests= input("What is your interest: ")

llm= ChatGoogleGenerativeAI(
    model="gemini-3-flash-preview",
    google_api_key=userdata.get('GOOGLE')
)

Day_p= PromptTemplate(
    template= " Must generate the plan as: Day 1, Day 2 … up to Day{number_of_days} and also take care of {city},{travel_style} and {interests}",
    input_variables= ["city", "number_of_days", "travel_style", "interests"]
)

time_blocks= PromptTemplate(
    template= "Each day of {Day_p} must contain: Morning, Afternoon, Evening,",
    input_variables= ["Day_p"]
)

budget_saving= PromptTemplate(
    template= '''At the end of each day,we required one practical money-saving suggestion for {time_blocks}, and output must be informat
                 Day 1
                 Morning: ...
                 Afternoon: ...
                 Evening: ...
                 Budget Tip: ...
                 Day 2

                 Morning: ...
                 Afternoon: ...
                 Evening: ...
                 Budget Tip: ...
                 (continue till all days) like that ''',

   input_variables= ["time_blocks"]
)

chain= Day_p | llm | StrOutputParser() | time_blocks | llm | StrOutputParser() | budget_saving | llm | StrOutputParser()

response= chain.invoke({"city": city,"number_of_days":number_of_days, "travel_style":travel_style, "interests": interests})

print(response)
