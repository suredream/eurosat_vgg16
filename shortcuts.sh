# zdown <number> <outname>
# new <repo_name>
# clone <repo_name>
# __ git status
# sync <repo> <msg>
# s3push <msg>
# s3pull 
# s3clone <repo>
# s3put_data team/person get

CFGFILE=~/.gitconfig
if [ ! -f "$FILE" ]; then #colab
    git config --global user.email "junxiong360@gmail.com"
    git config --global user.name "Jun Xiong"
fi

token=$(<~/.gtoken)
s3team=s3://com.climate.production.users/teams/geospatial/DerivedData/Enviornment/projects/
s3person=s3://com.climate.production.users/people/jun.xiong/repo_data/
github=github.com/suredream/
gitlab=s3://com.climate.production.users/people/jun.xiong/gitlab/
ztoken=4Y3bsknsNL2nmIscv52t5tfzrvAfHZpx626U12KUphCO1DaFwwOdAVWpBXhU
zdown() {
    curl --cookie /zenodo-cookies.txt "https://zenodo.org/record/$1?token=$ztoken" >/dev/null
    curl --cookie zenodo-cookies.txt "https://zenodo.org/record/$1/files/$2?download=1" --output $2
}
new () {
    url=$token@$github"flow".git
    git clone $1
    rm -fr $1/.git
    cd $1
    git init
}
clone () {
    url=$token@$github$1.git
    git clone $url tmp && mv tmp/.git . && rm -rf tmp && git reset --hard
}
sync () {
    # url=$token@$github$1.git
    git pull
    git add -u
    git commit -m "${@:1}"
    git push # $url
}
__ () {
    git status
}
s3push () {
    dest=`basename $PWD`
    git add .
    git commit -m "${@:1}"
    git bundle create $dest.bundle HEAD master;  
    aws s3 cp $dest.bundle $gitlab;  
}
s3pull () {
    cd $1
    git stash
    aws s3 cp $gitlab$1.bundle .
    git pull $1.bundle -f
}
s3clone () {
    aws s3 cp $gitlab$1.bundle .
    git clone $1.bundle
}
s3put_data () {
    dest=$(basename -s .git `git config --get remote.origin.url`)
    if [ "$1" = "team" ]; then
        folder=$s3team
    else
        folder=$s3person
    fi
    if [[ "$*" == *"get"* ]]; then # get
        aws s3 sync $folder$dest/data data
    else # put
        aws s3 sync data $folder$dest/data
    fi
}
