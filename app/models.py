from pydantic import BaseModel
from typing import List, Dict, Optional


class BasketballFrontAnalysisSummary(BaseModel):
    frontElbowPrediction: Optional[Dict[str, float]]
    frontLegsPrediction: Optional[Dict[str, float]]


class AnalysisDocument(BaseModel):
    analysisCategory: str
    analysisName: str
    analysisSummary: BasketballFrontAnalysisSummary
    owner: str
    sourceVideoPath: str
    outputVideoPath: str