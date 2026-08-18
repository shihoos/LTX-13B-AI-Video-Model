def sync_repo(
    url,
    path,
    commit,
):
    """
    Synchronize a repository to one exact locked commit.

    Handles both normal and shallow repositories.
    This is important because an old pinned commit may
    be unreachable from a depth-1 clone until the repository
    is explicitly unshallowed.
    """

    if (
        path.exists()
        and not (
            path
            / ".git"
        ).exists()
    ):
        shutil.rmtree(
            path
        )

    if not path.exists():

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        run(
            [
                "git",
                "clone",
                url,
                str(path),
            ]
        )

    # Detect whether the existing checkout is shallow.
    shallow_result = subprocess.run(
        [
            "git",
            "rev-parse",
            "--is-shallow-repository",
        ],
        cwd=path,
        capture_output=True,
        text=True,
        check=True,
    )

    is_shallow = (
        shallow_result.stdout.strip()
        == "true"
    )

    if is_shallow:

        print(
            f"Repository is shallow: {path.name}"
        )

        run(
            [
                "git",
                "fetch",
                "--unshallow",
                "origin",
            ],
            cwd=path,
        )

    else:

        run(
            [
                "git",
                "fetch",
                "--all",
                "--tags",
                "--prune",
            ],
            cwd=path,
        )

    # Make sure tags/remote references are refreshed
    # after unshallowing as well.
    run(
        [
            "git",
            "fetch",
            "--tags",
            "--prune",
            "origin",
        ],
        cwd=path,
    )

    run(
        [
            "git",
            "checkout",
            "--force",
            commit,
        ],
        cwd=path,
    )

    actual = git_current(
        path
    )

    if actual != commit:

        raise RuntimeError(
            "Git revision mismatch:\n"
            f"Path:     {path}\n"
            f"Expected: {commit}\n"
            f"Actual:   {actual}"
        )

    print(
        f"✅ {path.name}: {actual}"
    )
