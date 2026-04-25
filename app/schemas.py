from pydantic import BaseModel


class HPI(BaseModel):
    onset: str
    location: str
    duration: str
    character: str
    severity: str
    aggravating: str
    relieving: str


class ClinicalBrief(BaseModel):
    chief_complaint: str
    hpi: HPI
    ros: dict[str, list[str]]
    generated_at: str
