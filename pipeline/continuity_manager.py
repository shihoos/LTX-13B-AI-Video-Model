class ContinuityManager:
    """
    Maintains relationships between shots.
    """

    def link_shots(
        self,
        shots: list,
    ) -> list:

        for index, shot in enumerate(shots):

            if index > 0:

                shot.previous_shot = (
                    shots[index - 1]
                    .shot_id
                )

            if index < len(shots) - 1:

                shot.next_shot = (
                    shots[index + 1]
                    .shot_id
                )

        return shots

    def build_context(
        self,
        previous_shot,
        current_shot,
    ) -> dict:

        context = {
            "previous_shot_id": None,
            "previous_location": None,
            "previous_characters": [],
            "continuity_notes": "",
        }

        if previous_shot is not None:

            context[
                "previous_shot_id"
            ] = previous_shot.shot_id

            context[
                "previous_location"
            ] = previous_shot.location

            context[
                "previous_characters"
            ] = previous_shot.characters

            context[
                "continuity_notes"
            ] = previous_shot.continuity_notes

        context[
            "current_shot_id"
        ] = current_shot.shot_id

        return context
