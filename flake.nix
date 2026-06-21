{
  description = "The City of Numeria — meetup document (LaTeX) and notebook";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs =
    { nixpkgs, ... }:
    let
      systems = [
        "x86_64-linux"
        "aarch64-linux"
        "x86_64-darwin"
        "aarch64-darwin"
      ];
      forAllSystems =
        f: nixpkgs.lib.genAttrs systems (system: f nixpkgs.legacyPackages.${system});

      # Only the LaTeX packages main.tex actually pulls in, plus latexmk.
      texEnvFor =
        pkgs:
        pkgs.texlive.combine {
          inherit (pkgs.texlive)
            scheme-basic
            latexmk
            geometry
            amsmath
            amsfonts # provides amssymb
            booktabs
            tools # provides array.sty
            enumitem
            xcolor
            hyperref
            microtype
            fancyhdr
            titlesec
            lm # Latin Modern fonts
            cm-super # Type1 CM fonts incl. TS1 (tcrm*)
            ;
        };

      pythonFor =
        pkgs:
        pkgs.python3.withPackages (
          ps: with ps; [
            numpy
            pandas
            matplotlib
            ipython
            jupyter
            notebook
          ]
        );
    in
    {
      packages = forAllSystems (pkgs: {
        default = pkgs.stdenvNoCC.mkDerivation {
          pname = "numeria-doc";
          version = "1.0";
          src = ./.;

          nativeBuildInputs = [ (texEnvFor pkgs) ];

          buildPhase = ''
            runHook preBuild
            export HOME=$TMPDIR
            latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
            runHook postBuild
          '';

          installPhase = ''
            runHook preInstall
            install -Dm644 main.pdf $out/main.pdf
            runHook postInstall
          '';
        };
      });

      devShells = forAllSystems (pkgs: {
        default = pkgs.mkShell {
          packages = [
            (texEnvFor pkgs)
            (pythonFor pkgs)
          ];
        };
      });
    };
}
