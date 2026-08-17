class ContinuityManager:

    """
    Maintains continuity across the complete
    production, including scene boundaries.
    """

    def link_shots(
        self,
        shots: list,
        previous_shot=None,
    ) -> list:

        if not shots:
            return []

        for index, shot in enumerate(
            shots
        ):

            if index > 0:

                shot.previous_shot = (
                    shots[
                        index - 1
                    ].shot_id
                )

            elif previous_shot is not None:

                shot.previous_shot = (
                    previous_shot.shot_id
                )

            else:

                shot.previous_shot = None

            if index < len(shots) - 1:

                shot.next_shot = (
                    shots[
                        index + 1
                    ].shot_id
                )

            else:

                shot.next_shot = None

        return shots

    def connect_previous_scene(
        self,
        previous_shot,
        current_shots: list,
    ):

        if (
            previous_shot is None
            or not current_shots
        ):
            return

        previous_shot.next_shot = (
            current_shots[0].shot_id
        )

    def build_context(
        self,
        previous_shot=None,
    ) -> str:

        if previous_shot is None:
            return ""

        parts = []

        parts.append(
            "Previous shot ID: "
            f"{previous_shot.shot_id}"
        )

        if previous_shot.location:

            parts.append(
                "Previous location: "
                f"{previous_shot.location}"
            )

        if previous_shot.characters:

            parts.append(
                "Characters present: "
                + ", ".join(
                    previous_shot.characters
                )
            )

        if previous_shot.action:

            parts.append(
                "Previous action: "
                f"{previous_shot.action}"
            )

        if previous_shot.camera_shot:

            parts.append(
                "Previous camera shot: "
                f"{previous_shot.camera_shot}"
            )

        if previous_shot.lighting:

            parts.append(
                "Previous lighting: "
                f"{previous_shot.lighting}"
            )

        if previous_shot.mood:

            parts.append(
                "Previous mood: "
                f"{previous_shot.mood}"
            )

        if previous_shot.continuity_notes:

            parts.append(
                "Continuity requirements: "
                f"{previous_shot.continuity_notes}"
            )

        return "\n".join(
            parts
        )

    def apply_scene_continuity(
        self,
        shots: list,
        previous_shot=None,
    ) -> list:

        if not shots:
            return []

        if previous_shot is not None:

            context = (
                self.build_context(
                    previous_shot
                )
            )

            first_shot = shots[0]

            if (
                first_shot.continuity_notes
            ):

                first_shot.continuity_notes = (
                    context
                    + "\n"
                    + first_shot.continuity_notes
                )

            else:

                first_shot.continuity_notes = (
                    context
                )

        self.link_shots(
            shots,
            previous_shot=previous_shot,
        )

        self.connect_previous_scene(
            previous_shot,
            shots,
        )

        return shots
