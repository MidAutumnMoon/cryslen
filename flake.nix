{
  description = "cryslen (cy): minimal wayland image/comic pager";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable-small";

  outputs = { self, nixpkgs }:
    let
      inherit (nixpkgs) lib;
      systems = [ "x86_64-linux" ];
      forAllSystems = nixpkgs.lib.genAttrs systems;
    in
    {
      devShells = forAllSystems (system:
        let pkgs = nixpkgs.legacyPackages.${system}; in {
          default = pkgs.mkShell {
            packages = [
              # toolchain to build the native deps (dav1d via libavif-sys)
              pkgs.cmake
              pkgs.nasm
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
            nativeBuildInputs = [
              pkgs.cmake
              pkgs.nasm
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
