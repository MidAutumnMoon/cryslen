{
  description = "cryslen (cy): minimal wayland image/comic pager";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable-small";

  outputs = { self, nixpkgs }:
    let
      inherit (nixpkgs) lib;
      systems = [ "x86_64-linux" ];
      forAllSystems = nixpkgs.lib.genAttrs systems;
      # aom's cmake configure greps `nasm -hf` for "-Ox", which nasm 3.x
      # dropped from the help text though the flag works.
      nasm-aom-shim = pkgs: pkgs.writeShellScriptBin "nasm" ''
        if [ "$1" = "-hf" ]; then
          ${pkgs.nasm}/bin/nasm "$@"
          echo "  -Ox                     multipass optimization (shim)"
        else
          exec ${pkgs.nasm}/bin/nasm "$@"
        fi
      '';
    in
    {
      devShells = forAllSystems (system:
        let pkgs = nixpkgs.legacyPackages.${system}; in {
          default = pkgs.mkShell {
            packages = [
              # toolchain to build the optional native deps (dav1d via libavif-sys)
              pkgs.cmake
              (nasm-aom-shim pkgs)
              pkgs.meson
              pkgs.ninja
              pkgs.perl
              pkgs.clang
              pkgs.pkg-config
            ];
            env.LD_LIBRARY_PATH = lib.makeLibraryPath (with pkgs; [
              wayland
              libxkbcommon
              libGL
              libinput
            ]);
          };
        });

      packages = forAllSystems (system:
        let pkgs = nixpkgs.legacyPackages.${system}; in {
          default = pkgs.rustPlatform.buildRustPackage {
            pname = "cryslen";
            version = "0.1.0";
            src = self;
            cargoLock.lockFile = ./Cargo.lock;
            buildFeatures = [ "avif" "jxl" ];
            nativeBuildInputs = [
              pkgs.cmake
              (nasm-aom-shim pkgs)
              pkgs.meson
              pkgs.ninja
              pkgs.perl
              pkgs.pkg-config
            ];
            meta.mainProgram = "cy";
          };
        });
    };
}
