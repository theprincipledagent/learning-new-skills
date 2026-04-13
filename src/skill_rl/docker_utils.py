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

    def ensure_images(self, actor_image: str, actor_dir: str) -> None:
        """Build actor image if not present."""
        try:
            self.client.images.get(actor_image)
            print(f"Image {actor_image} already exists")
        except ImageNotFound:
            self.build_image(actor_dir, actor_image)

    def run_container(
        self,
        image: str,
        command: str | list[str] | None = None,
        env: dict[str, str] | None = None,
        volumes: dict[str, dict] | None = None,
        timeout: int = 600,
        extra_hosts: dict[str, str] | None = None,
    ) -> tuple[int, str, str]:
        """Run a container to completion. Always cleans up.

        Returns (exit_code, stdout, stderr).
        """
        container = None
        try:
            kwargs = dict(
                image=image,
                environment=env or {},
                volumes=volumes or {},
                detach=True,
                stdin_open=False,
                tty=False,
            )
            if command:
                kwargs["command"] = command
            if extra_hosts:
                kwargs["extra_hosts"] = extra_hosts

            container = self.client.containers.run(**kwargs)
            result = container.wait(timeout=timeout)
            exit_code = result["StatusCode"]
            stdout = container.logs(stdout=True, stderr=False).decode("utf-8", errors="replace")
            stderr = container.logs(stdout=False, stderr=True).decode("utf-8", errors="replace")
            if stderr:
                print(f"  Container stderr: {stderr[:2000]}")
            return exit_code, stdout, stderr
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
