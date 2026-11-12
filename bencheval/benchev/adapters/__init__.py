def get_adapter(name):
    if name == "gui_aima_qwen35":
        from benchev.adapters.gui_aima_qwen35 import GuiAimaQwen35
        return GuiAimaQwen35()
    if name == "gui_aima_upstream":
        from benchev.adapters.gui_aima_upstream import GuiAimaUpstream
        return GuiAimaUpstream()
    if name == "text_coords":
        from benchev.adapters.text_coords import TextCoords
        return TextCoords()
    if name == "gui_actor":
        from benchev.adapters.gui_actor import GuiActor
        return GuiActor()
    if name == "hf_chat":
        from benchev.adapters.hf_chat import HfChat
        return HfChat()
    if name == "deepseek_vl2":
        from benchev.adapters.deepseek_vl2 import DeepseekVl2
        return DeepseekVl2()
    if name == "cogagent":
        from benchev.adapters.cogagent import CogAgent
        return CogAgent()
    if name == "seeclick":
        from benchev.adapters.seeclick import SeeClick
        return SeeClick()
    raise ValueError(f"adapter {name!r} not implemented")
