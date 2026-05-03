{
  description = "Development shell with uv (Python package manager)";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system;
                                config.allowUnfree = true; };
        python = pkgs.python312;
      in {
        devShells.default = pkgs.mkShell {
          name = "uv-dev-shell";
          buildInputs = [
            pkgs.uv
            pkgs.git
          ];

          shellHook = ''

            # Only activate venv if it exists, don’t auto-create
            if [ -d ".venv" ]; then
              source .venv/bin/activate
            else
              echo "⚠️  No .venv found. Run: uv venv"
            fi
          '';
        };
      }
    );
}
