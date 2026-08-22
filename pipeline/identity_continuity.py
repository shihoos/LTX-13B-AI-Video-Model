from __future__ import annotations


class IdentityContinuity:
    """
    H3 replacement for the identity-preservation responsibility
    that previously depended on LTX-specific conditioning.

    This layer does not pretend to be IC-LoRA.

    It converts the existing structured character identity data
    into deterministic H3 conditioning text and negative guards.
    """

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

        return locks

    @staticmethod
    def build_reference_bindings(
        image_paths: list[str],
        character_bindings: dict[str, list[str]],
    ) -> list[str]:
        reverse = {}

        for character_name, paths in character_bindings.items():
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

            if names:
                result.append(
                    (
                        f"<Picture {index}> is the canonical "
                        f"identity reference for "
                        f"{', '.join(names)}. "
                        "Preserve that character's face, hair, "
                        "body proportions and distinctive features."
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
                "SHOT DIRECTION: "
                + visual_prompt.strip()
            )

        positive = " ".join(
            item
            for item in positive_parts
            if item.strip()
        )

        negative = (
            negative_prompt.strip()
            + ", identity drift, different face, "
              "different hairstyle, different hairline, "
              "different body proportions, altered skin tone, "
              "different distinctive marks, facial deformation, "
              "duplicate person, extra limbs, anatomy drift"
        ).strip(" ,")

        return positive, negative
