class ContinuityManager:
    """
    Maintains continuity between generated shots.
    """

    def link_shots(
        self,
        shots: list,
    ) -> list:

        for index, shot in enumerate(shots):

            if index > 0:
                shot.previous_shot = (
                    shots[index - 1].shot_id
                )
            else:
                shot.previous_shot = None

            if index < len(shots) - 1:
                shot.next_shot = (
                    shots[index + 1].shot_id
                )
            else:
                shot.next_shot = None

        return shots

    def build_context(
        self,
        previous_shot=None,
    ) -> str:
        """
        Build continuity information for planning
        the next shot or scene.
        """

        if previous_shot is None:
            return ""

        parts = []

        if previous_shot.location:
            parts.append(
                f"Previous location: "
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
                f"Previous action: "
                f"{previous_shot.action}"
            )

        if previous_shot.continuity_notes:
            parts.append(
                "Continuity requirements: "
                f"{previous_shot.continuity_notes}"
            )

        if not parts:
            return ""

        return "\n".join(parts)

    def apply_scene_continuity(
        self,
        shots: list,
        previous_shot=None,
    ) -> list:
        """
        Add continuity information from the previous
        scene to the first shot of the current scene,
        then link all shots in the current scene.
        """

        if not shots:
            return []

        if previous_shot is not None:

            context = self.build_context(
                previous_shot
            )

            first_shot = shots[0]

            if first_shot.continuity_notes:
                first_shot.continuity_notes = (
                    context
                    + "\n"
                    + first_shot.continuity_notes
                )
            else:
                first_shot.continuity_notes = (
                    context
                )

        return self.link_shots(
            shots
        )
