"use strict";

class UserStorage {
    static #users = {
        id: ["hwuihyun", "dh", "sol"],
        psword: ["1234", "1234", "123456"],
        name: ["조휘현", "이동현", "박참솔"]
    };

    static getUsers(...fields) {
        const users = this.#users;
        const newUsers = fields.reduce((newUsers, field) => {
            if (users.hasOwnProperty(field)) {
                newUsers[field] = users[field];
            }
            return newUsers;
        }, {});
        return newUsers;
    }
}

module.exports = UserStorage;