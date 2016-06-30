/*
 * This file is part of INSPIRE.
 * Copyright (C) 2016 CERN.
 *
 * INSPIRE is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License as
 * published by the Free Software Foundation; either version 2 of the
 * License, or (at your option) any later version.
 *
 * INSPIRE is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with INSPIRE; if not, write to the Free Software Foundation, Inc.,
 * 59 Temple Place, Suite 330, Boston, MA 02111-1307, USA.
 *
 * In applying this license, CERN does not
 * waive the privileges and immunities granted to it by virtue of its status
 * as an Intergovernmental Organization or submit itself to any jurisdiction.
 */
(function (angular) {
  /**
   * HoldingPenRecordService allows for the update of a record server
   * side through a post of the record JSON.
   */
  angular.module('holdingpen.services', [])
    .factory("HoldingPenRecordService", ["$http",
      function ($http) {
        return {
          /**
           * getRecord
           * @param vm
           * @param workflowId
           */
          getRecord: function (vm, workflowId) {
            $http.get('/api/holdingpen/' + workflowId).then(function (response) {
              vm.record = response.data;
              if(vm.record._workflow.data_type == 'authors') {
                $('#breadcrumb').html(vm.record.metadata.name.value);
              } else {
                $('#breadcrumb').html(vm.record.metadata.breadcrumb_title);
              }


            }).catch(function (value) {
              vm.ingestion_complete = false;
              alert(value);
            });
          },

          updateRecord: function (vm, workflowId) {
            $http.post('/api/holdingpen/' + workflowId + '/action/edit', vm.record).then(function (response) {
              vm.saved = true;
              vm.update_ready = false;
            }).catch(function (value) {
              vm.saved = false;
              vm.update_ready = true;
              alert('Sorry, an error occurred when saving. Please try again.');
            });
          },

          setDecision: function (vm, workflowId, decision) {
            var data = JSON.stringify({
              'value': decision
            });
            $http.post('/api/holdingpen/' + workflowId + '/action/resolve', data).then(function (response) {
              vm.ingestion_complete = true;
              var record = vm.record;
              if (!record) record = vm;
              record._extra_data.user_action = decision;
              record._extra_data._action = null;

            }).catch(function (value) {
              vm.error = value;
            });
          },

          setBatchDecision: function (records, selected_record_ids, decision) {

            var data = JSON.stringify({
              'value': decision,
              'object_ids': selected_record_ids,
              'action': 'resolve'
            });

            $http.post('/api/holdingpen/action/resolve', data).then(function (response) {
              // Should provide a quicker way to access the records
              for (var record in records) {
                if(selected_record_ids.indexOf(+records[record]._id) !== -1 ) {
                  var record_obj = records[record]._source;
                  record_obj._extra_data.user_action = decision;
                  record_obj._extra_data._action = null;
                }
              }

              selected_record_ids = [];

            }).catch(function (value) {
              alert(value);
            });
          },


          deleteRecord: function (vm, workflowId) {
            $http.delete('/api/holdingpen/' + workflowId, vm.record).then(function (response) {
              vm.ingestion_complete = true;
            }).catch(function (value) {
              alert(value);
              vm.ingestion_complete = false;
            });
          },

          resumeWorkflow: function (vm, workflowId) {
            $http.post('/api/holdingpen/' + workflowId + '/action/resume').then(function (response) {
              vm.workflow_flag = 'Workflow resumed';
            }).catch(function (value) {
              alert(value);
              vm.resumed = false;
            });
          },

          restartWorkflow: function (vm, workflowId) {
            $http.post('/api/holdingpen/' + workflowId + '/action/restart').then(function (response) {
              vm.workflow_flag = 'Workflow restarted';
            }).catch(function (value) {
              alert(value);
              vm.restarted = false;
            });
          }
        }
      }]
    );
}(angular));