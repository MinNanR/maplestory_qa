from backend.tool.dispatcher import dispatcher, ToolCallRequest
import backend.tool.tools_impl
import json

if __name__=='__main__':
    param = {
        "doc_id": "skill_arch-mage-ice-lightning-",
        "retrieval_texts": "Jupiter Thunder"
    }
    r = ToolCallRequest(tool_call_id="", name="retrieval_document", params=json.dumps(param))
    result = dispatcher.dispatch(r)
    print(result.content)