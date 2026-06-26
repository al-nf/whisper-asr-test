# whisper/flake.nix
{
  inputs.nixpkgs.url = "nixpkgs/nixos-unstable";
  
  outputs = { self, nixpkgs }: {
    devShells.x86_64-linux.default = let
      pkgs = import nixpkgs { 
        system = "x86_64-linux"; 
        config.allowUnfree = true;
      };
    in pkgs.mkShell {
      packages = [ pkgs.uv pkgs.ffmpeg ];
      LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
        pkgs.stdenv.cc.cc.lib
        pkgs.linuxPackages.nvidia_x11
        pkgs.ffmpeg
      ];
    };
  };
}
