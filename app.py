from fastapi import FastAPI
from pydantic import BaseModel
from query_langchain import load_qa_chain

app = FastAPI()

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str

@app.get("/")
def home():
    return {"message": "API running"}

@app.post("/ask", response_model=QueryResponse)
async def ask(req: QueryRequest):
    chain = load_qa_chain()

    if chain is None:
        return QueryResponse(answer="❌ QA chain failed to load. Check Azure env variables.")

    result = chain.invoke({"query": req.question})
    return QueryResponse(answer=result.get("result"))
