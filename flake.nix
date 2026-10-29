{
  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable-small";

  outputs = { self, nixpkgs }:
    let
      inherit (nixpkgs) lib;
      inherit (lib) makeLibraryPath;
      pkgs = nixpkgs.legacyPackages.x86_64-linux;

      dlopenLibs = with pkgs; [
        wayland
        libxkbcommon
        libglvnd
        fontconfig
      ];

      package = pkgs.rustPlatform.buildRustPackage {
        pname = "cryslen";
        version = "0.1.0";
        src = self;
        cargoLock.lockFile = ./Cargo.lock;

        nativeBuildInputs = with pkgs; [
          cmake
          meson
          ninja
          nasm
          makeBinaryWrapper
        ];
        dontUseNinjaBuild = true;
        dontUseNinjaInstall = true;

        postInstall = ''
          install -Dm644 assets/cryslen.desktop -t $out/share/applications
          mkdir -p $out/share/icons
          cp -r assets/icons/hicolor $out/share/icons/

          wrapProgram $out/bin/cy \
            --suffix LD_LIBRARY_PATH : ${makeLibraryPath dlopenLibs}
        '';

        meta = {
          homepage = "https://github.com/MidAutumnMoon/cryslen";
          mainProgram = "cy";
          platforms = lib.platforms.linux;
          license = lib.licenses.gpl3Plus;
        };
      };
    in
    {
      packages.x86_64-linux.default = package;

      devShells.x86_64-linux.default = pkgs.mkShell {
        inputsFrom = [ package ];
        packages = [ pkgs.clang ];
        env.LD_LIBRARY_PATH = makeLibraryPath dlopenLibs;
      };
    };
}
