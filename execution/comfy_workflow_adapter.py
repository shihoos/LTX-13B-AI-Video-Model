@staticmethod
def set_input_video(
    workflow: dict,
    filename: str,
):
    for node in workflow.values():

        if node.get("class_type") != "VHS_LoadVideo":
            continue

        inputs = node.setdefault(
            "inputs",
            {},
        )

        if "video" in inputs:
            inputs["video"] = filename
            return

    raise RuntimeError(
        "Workflow has no VHS_LoadVideo node "
        "with a writable 'video' input."
    )
@staticmethod
def set_input_video(
    workflow: dict,
    filename: str,
):
    for node in workflow.values():

        if node.get("class_type") != "VHS_LoadVideo":
            continue

        inputs = node.setdefault(
            "inputs",
            {},
        )

        if "video" in inputs:
            inputs["video"] = filename
            return

    raise RuntimeError(
        "Workflow has no VHS_LoadVideo node "
        "with a writable 'video' input."
    )
