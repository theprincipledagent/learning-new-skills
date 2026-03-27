"""Docker SDK helpers for running containers."""

import json
from pathlib import Path

import docker
from docker.errors import BuildError, ContainerError, ImageNotFound


class DockerManager:
    def __init__(self):
        self.client = docker.from_env()

    def build_image(self, build_dir: str, tag: str) -> None:
        """Build a Docker image from a directory."""
        print(f"Building Docker image {tag} from {build_dir}...")
        try:
            self.client.images.build(
                path=build_dir,
                tag=tag,
                rm=True,
            )
            print(f"  Built {tag}")
        except BuildError as e:
            print(f"  Build failed for {tag}:")
            for chunk in e.build_log:
                if "stream" in chunk:
                    print(f"    {chunk['stream'].strip()}")
            raise

    def ensure_images(self, actor_image: str, llm_image: str,
                      actor_dir: str, llm_dir: str) -> None:
        """Build actor and llm images if not present."""
        for tag, build_dir in [(actor_image, actor_dir), (llm_image, llm_dir)]:
            try:
                self.client.images.get(tag)
                print(f"Image {tag} already exists")
            except ImageNotFound:
                self.build_image(build_dir, tag)

    def run_container(
        self,
        image: str,
        command: str,
        env: dict[str, str] | None = None,
        volumes: dict[str, dict] | None = None,
        timeout: int = 600,
    ) -> tuple[int, str]:
        """Run a container to completion. Always cleans up.

        Returns (exit_code, stdout).
        """
        container = None
        try:
            container = self.client.containers.run(
                image=image,
                command=command,
                environment=env or {},
                volumes=volumes or {},
                detach=True,
                stdin_open=False,
                tty=False,
            )
            result = container.wait(timeout=timeout)
            exit_code = result["StatusCode"]
            stdout = container.logs(stdout=True, stderr=False).decode("utf-8", errors="replace")
            stderr = container.logs(stdout=False, stderr=True).decode("utf-8", errors="replace")
            if stderr:
                print(f"  Container stderr: {stderr[:2000]}")
            return exit_code, stdout
        except Exception as e:
            if container:
                try:
                    container.kill()
                except Exception:
                    pass
            raise RuntimeError(f"Container execution failed: {e}") from e
        finally:
            if container:
                try:
                    container.remove(force=True)
                except Exception:
                    pass

    def run_container_with_volume_io(
        self,
        image: str,
        input_data: dict,
        work_dir: Path,
        env: dict[str, str] | None = None,
        timeout: int = 600,
        prompt_mount: Path | None = None,
    ) -> dict | None:
        """Run a container with JSON I/O via volume mount.

        Writes input_data to work_dir/input.json, mounts work_dir at /work
        and optionally mounts prompt_mount at /prompts/system.txt.
        Reads output from work_dir/output.json.
        """
        work_dir.mkdir(parents=True, exist_ok=True)
        input_path = work_dir / "input.json"
        input_path.write_text(json.dumps(input_data, indent=2))

        volumes = {
            str(work_dir.resolve()): {"bind": "/work", "mode": "rw"},
        }
        if prompt_mount:
            volumes[str(prompt_mount.resolve())] = {
                "bind": "/prompts/system.txt",
                "mode": "ro",
            }

        exit_code, stdout = self.run_container(
            image=image,
            command="",  # Uses default entrypoint
            env=env,
            volumes=volumes,
            timeout=timeout,
        )

        output_path = work_dir / "output.json"
        if output_path.exists():
            return json.loads(output_path.read_text())

        print(f"  Warning: No output.json found. Exit code: {exit_code}")
        print(f"  stdout: {stdout[:2000]}")
        return None
