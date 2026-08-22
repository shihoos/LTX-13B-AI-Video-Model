from __future__ import annotations


class IdentityContinuity:

    @staticmethod
    def build_locks(
        characters: list,
        shot_characters: list[str],
    ) -> list[str]:

        by_name = {
            character.name.lower(): character
            for character in characters
        }

        locks = []

        for name in shot_characters:

            character = by_name.get(
                str(name).lower()
            )

            if character is None:
                continue

            locks.append(
                character.identity_lock_text()
            )

            state_text = (
                character.story_state_text()
            )

            if state_text:
                locks.append(
                    state_text
                )

        return locks

    @staticmethod
    def build_reference_bindings(
        image_paths: list[str],
        character_bindings: dict[str, list[str]],
    ) -> list[str]:

        reverse = {}

        for character_name, paths in (
            character_bindings.items()
        ):

            for path in paths:
                reverse.setdefault(
                    str(path),
                    [],
                ).append(
                    character_name
                )

        result = []

        for index, image_path in enumerate(
            image_paths,
            start=1,
        ):

            names = reverse.get(
                str(image_path),
                [],
            )

            if not names:
                continue

            result.append(
                (
                    f"<Picture {index}> is the "
                    f"canonical visual identity reference "
                    f"for {', '.join(names)}. "
                    "Use this reference specifically for "
                    "face, hair, hairline, body structure, "
                    "body proportions and stable identity "
                    "features."
                )
            )

        return result

    @staticmethod
    def merge(
        visual_prompt: str,
        locks: list[str],
        bindings: list[str],
        negative_prompt: str,
    ) -> tuple[str, str]:

        positive_parts = []

        positive_parts.extend(
            bindings
        )

        positive_parts.extend(
            locks
        )

        if visual_prompt.strip():
            positive_parts.append(
                visual_prompt.strip()
            )

        positive = "\n".join(
            part
            for part in positive_parts
            if str(part).strip()
        )

        # H3 does not expose a conventional negative
        # conditioning socket. Keep this as a textual
        # production guard for our planner/export layer.
        negative = (
            str(
                negative_prompt or ""
            ).strip()
        )

        if negative:
            negative += "\n"

        negative += (
            "Avoid identity drift, altered face geometry, "
            "different hairstyle, different hairline, "
            "different body proportions, altered skin tone, "
            "facial deformation, extra limbs, duplicate person."
        )

        return (
            positive,
            negative,
        )
