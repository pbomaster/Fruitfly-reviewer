"""Generic abstract/body headings for full extracted documents; never filter text."""
import re

START={'ABSTRACT','SUMMARY'}
END={'INTRODUCTION','RESULTS','DISCUSSION','METHODS','MATERIALS AND METHODS','RESULTS AND DISCUSSION','STAR METHODS','REFERENCES'}
def document_sentences(pages):
    section='body'; blocks=[]; current=[]; block_section=section; block_page=1
    def flush():
        if current:
            text=re.sub(r'\s+',' ','\n'.join(current)).strip()
            if text: blocks.append((block_page,block_section,text))
            current.clear()
    for page,raw in enumerate(pages,1):
        flush(); block_page=page
        for line in raw.splitlines():
            heading=re.sub(r'[^A-Z ]','',line.strip()).strip()
            change=None
            if line.strip()==line.strip().upper():
                if heading in START: change='abstract'
                elif heading in END: change='body'
            if change is not None:
                flush(); section=change; block_section=section
            current.append(line)
    flush(); result=[]
    for page,section,text in blocks:
        for s in re.split(r'(?<=[.!?])\s+(?=[A-Z])',text):
            result.append({'page':page,'section':section,'text':s})
    assert re.sub(r'\s+','',''.join(pages))==re.sub(r'\s+','',''.join(x['text'] for x in result))
    return result
