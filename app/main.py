import shutil
from fastapi import FastAPI
from app import analyses, models


app = FastAPI()
    

@app.post("/analyze_basketball_front/")
def analyze_basketball_front(analysis_document: models.AnalysisDocument):
    analysisSummary, outputVideoPath = analyses.basketball_front(analysis_document.sourceVideoPath)
    analysisCategory = analysis_document.analysisCategory
    analysisName = analysis_document.analysisName
    owner = analysis_document.owner
    sourceVideoPath = analysis_document.sourceVideoPath

    responseDocument = models.AnalysisDocument(
        analysisCategory=analysisCategory,
        analysisName=analysisName,
        analysisSummary=analysisSummary,
        owner=owner,
        sourceVideoPath=sourceVideoPath,
        outputVideoPath=outputVideoPath,
    )

    return responseDocument