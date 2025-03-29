# To learn more about how to use Nix to configure your environment
# see: https://developers.google.com/idx/guides/customize-idx-env
{ pkgs, ... }: {
  # Which nixpkgs channel to use.
  channel = "stable-24.05"; # or "unstable"
  # Use https://search.nixos.org/packages to find packages
  packages = [
    # pkgs.go
    pkgs.aws-sam-cli
    pkgs.ruff
    pkgs.pre-commit
    pkgs.python311
    pkgs.python311Packages.pip
    pkgs.nodejs_20
    pkgs.commitizen
    pkgs.nodePackages.nodemon
    pkgs.docker
    pkgs.mlflow-server
    pkgs.gh
    pkgs.docker-compose
    pkgs.bleachbit
    pkgs.htop
    pkgs.neofetch
    pkgs.ghfetch
    pkgs.ipfetch
    pkgs.psutils
    pkgs.awscli2
    pkgs.zip
    pkgs.unzip
    pkgs.gnumake
    pkgs.nano
    pkgs.speedtest-cli
    pkgs.poetry
    pkgs.uv
    pkgs.iputils
  ];
  # Sets environment variables in the workspace
  services.docker.enable = true;
  env = {};
  idx = {
    # Search for the extensions you want on https://open-vsx.org/ and use "publisher.id"
    extensions = [
      # "vscodevim.vim"
      "EchoAPI.echoapi-for-vscode"
      "zhuangtongfa.material-theme"
      "GitHub.vscode-pull-request-github"
      "GitHub.vscode-github-actions"
      "ms-edgedevtools.vscode-edge-devtools"
      "Google.geminicodeassist"
      "shd101wyy.markdown-preview-enhanced"
      "charliermarsh.ruff"
      "PKief.material-icon-theme"
      "monokai.theme-monokai-pro-vscode"
      "lrstanley.excalidraw-editor"
      "hediet.vscode-drawio"
      "amazonwebservices.aws-toolkit-vscode"
      "OmriGM.codeium"
      # "rangav.vscode-thunder-client"
      "ms-azuretools.vscode-docker"
      "Codeium.codeium"
      # "Codium.codium"
      "augment.vscode-augment"
      "littlefoxteam.vscode-python-test-adapter"
      "luma.jupyter"
      "ms-toolsai.jupyter"
      "ms-python.python"
      # "lrstanley.excalidraw-editor"
      "franneck94.vscode-python-dev-extension-pack"
      "charliermarsh.ruff"
      "franneck94.vscode-python-config"
      "ms-python.debugpy"
      "ms-toolsai.jupyter-keymap"
      "ms-toolsai.jupyter-renderers"
      "ms-toolsai.vscode-jupyter-cell-tags"
      "ms-toolsai.vscode-jupyter-slideshow"
      "njpwerner.autodocstring"
      "njqdev.vscode-python-typehint"
      "tamasfe.even-better-toml"
      "hbenl.vscode-test-explorer"
      "ms-vscode.test-adapter-converter"
    ];
    # Enable previews
    previews = {
      enable = true;
      previews = {
        # web = {
        #   # Example: run "npm run dev" with PORT set to IDX's defined port for previews,
        #   # and show it in IDX's web preview panel
        #   command = ["npm" "run" "dev"];
        #   manager = "web";
        #   env = {
        #     # Environment variables to set for your server
        #     PORT = "$PORT";
        #   };
        # };
      };
    };
    
    # Workspace lifecycle hooks
    workspace = {
      # Runs when a workspace is first created
      onCreate = {
        # Example: install JS dependencies from NPM
        # npm-install = "npm install";
        # cdk-install = "npm install -g aws-cdk@latest";
        # Open editors for the following files by default, if they exist:
        # default.openFiles = [ ".idx/dev.nix" "README.md" ];
      };
      # Runs when the workspace is (re)started
      onStart = {
        # Example: start a background task to watch and re-build backend code
        # watch-backend = "npm run watch-backend";

      };
    };
  };
}
